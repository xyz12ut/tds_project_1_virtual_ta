#setting up chroma db for embeddings

#imports
import os
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_chroma import Chroma
import yaml
import shutil
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential

import config

# Constants
CHROMA_PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in" #base url for discourse
SUMMARY_MODEL = "gpt-3.5-turbo-16k"  # Using 16k context model for summaries



client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))  # for summarisation

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_weighted_summary(posts: List[Dict[str, Any]], weights: List[float]) -> str:
    """Generate a summary using OpenAI API that respects post weights."""
    # Prepare the prompt with weighted context
    prompt_parts = []
    for post, weight in zip(posts, weights):
        prefix = "[FACULTY] " if post.get("user_title") == "Course_faculty" else ""
        prompt_parts.append(
            f"{prefix}Weight: {weight:.1f}\n"
            f"Content: {post.get('cleaned_text', '')}\n"
            f"Engagement: {post.get('reply_count', 0)} replies, "
            f"{post.get('reaction_users_count', 0)} reactions\n"
        )
    
    context = "\n".join(prompt_parts)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes discussion threads while respecting the weight/importance of each post. Faculty posts should be emphasized."},
            {"role": "user", "content": f"Create a comprehensive summary of this discussion thread, giving appropriate weight to each post based on the provided weights. Focus on technical details for technical support topics.\n\n{context}"}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def calculate_post_weight(post: Dict[str, Any]) -> float:
    """Enhanced weight calculation with more nuanced factors."""
    weight = 0.0
    
    # Base metrics (normalized)
    weight += min(post.get("score", 0) / 10, 50)  # Cap at 50
    weight += min(post.get("reads", 0) / 100, 30)  # Cap at 30
    weight += post.get("trust_level", 0) * 15  # 0-4 → 0-60
    
    # Reactions get diminishing returns
    reactions = post.get("reaction_users_count", 0)
    weight += min(reactions * 5, 50) + (min(reactions, 5) * 2)  # Bonus for first 5
    
    # Author factors
    if post.get("user_title") == "Course_faculty":
        weight *= 1.8  # Strong faculty boost
    elif post.get("primary_group_name") == "ds-students":
        weight *= 0.9  # Slight student penalty
    
    # Content factors
    content = post.get("cleaned_text", "")
    if len(content.split()) > 50:  # Longer posts get boost
        weight *= 1.2
    if "?" in content:  # Questions get slight boost
        weight *= 1.1
    
    # Thread structure
    if post.get("post_number", 0) == 1:  # OP gets boost
        weight *= 1.3
    if post.get("accepted_answer", False):
        weight *= 2.5  # Strong accepted answer boost
    
    weight += post.get("post_number", 0)**1.5
    
    return weight

def process_topic_posts(topic_data: Dict[str, Any]) -> Document:
    """Process all posts in a topic into a single summarized Document."""
    posts = topic_data["posts"]
    weights = [calculate_post_weight(p) for p in posts]
    
    # Filter out very low weight posts (improves summary quality)
    filtered_data = [
        (p, w) for p, w in zip(posts, weights) 
        if w > 10 or p.get("user_title") == "Course_faculty"
    ]
    if not filtered_data:
        return None
        
    filtered_posts, filtered_weights = zip(*filtered_data)
    
    # Generate AI summary
    summary = generate_weighted_summary(filtered_posts, filtered_weights)
    
    topic_url = None
    if posts and "post_url" in posts[0]:
        url_parts = posts[0]["post_url"].split("/")
        topic_url = "/".join(url_parts[:-1])  # Remove last segment (post number)
        topic_url = "/".join(topic_url)  # Reconstruct topic URL
        topic_url= BASE_URL + topic_url
    
    metadata = {
        "source": topic_data["source_file"],
        "topic_id": str(topic_data["topic_id"]),
        "topic": topic_data["topic"],
        "topic_slug": topic_data["topic_slug"],
        "post_count": len(posts),
        "faculty_post_count": sum(1 for p in posts if p.get("user_title") == "Course_faculty"),
        "total_weight": sum(weights),
        "document_type": "discourse_topic",
        "created_at": min(p.get("created_at", "") for p in posts),  # Earliest post date
        "updated_at": max(p.get("updated_at", "") for p in posts),   # Latest update
        "original_url" : topic_url
    }
    
    return Document(page_content=summary, metadata=metadata)

def load_discourse_json_files(path: str, recursive: bool = False) -> List[Document]:
    """Load Discourse JSON files and return both topic summaries and faculty posts."""
    topic_documents = []
    faculty_documents = []
    
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process posts
                topics = {}
                
                for post in data.get("post_stream", {}).get("posts", []):
                    # Extract faculty posts for individual embedding
                    if post.get("user_title") == "Course_faculty":
                        faculty_doc = create_faculty_post_document(post, file_path)
                        faculty_documents.append(faculty_doc)
                    
                    # Group posts by topic for summaries
                    topic_id = post.get("topic_id")
                    if topic_id:
                        if topic_id not in topics:
                            topics[topic_id] = {
                                "posts": [],
                                "topic": post.get("topic", "uncategorized"),
                                "topic_slug": post.get("topic_slug", ""),
                                "source_file": file_path,
                                "topic_id": topic_id
                            }
                        topics[topic_id]["posts"].append(post)
                
                
                # Create topic summaries
                for topic_data in topics.values():
                    doc = process_topic_posts(topic_data)
                    if doc:
                        topic_documents.append(doc)
                        
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        if not recursive:
            break
    
    print(f"Processed {len(topic_documents)} topics and {len(faculty_documents)} faculty posts")
    return topic_documents + faculty_documents


#separate embeddings for posts by faculties
def create_faculty_post_document(post: Dict[str, Any], source_path: str) -> Document:

    
    url_parts = post.get("post_url", "").split("/")
    topic_url = "/".join(url_parts[:-1])  # Remove last segment (post number)
    topic_url = "".join(topic_url)  # Reconstruct topic URL
    topic_url = BASE_URL + topic_url

    """Create a Document for an individual faculty post with rich metadata."""
    metadata = {
        "source": source_path,
        "post_id": str(post.get("id")),
        "topic_id": str(post.get("topic_id")),
        "topic": post.get("topic", "uncategorized"),
        "topic_slug": post.get("topic_slug", ""),
        "created_at": post.get("created_at", ""),
        "updated_at": post.get("updated_at", ""),
        "post_number": post.get("post_number", 0),
        "document_type": "faculty_post",
        "engagement_score": calculate_post_weight(post),
        "reply_count": post.get("reply_count", 0),
        "reactions": post.get("reaction_users_count", 0),
        "reads": post.get("reads", 0),
        "original_url": topic_url
    }
    
    # Enhance the content with context
    content = (
        f"[Faculty Post in '{metadata['topic']}']\n"
        f"Post #{metadata['post_number']} (Score: {metadata['engagement_score']:.1f})\n\n"
        f"{post.get('cleaned_text', '')}\n\n"
        f"Context: {post.get('topic', '')} discussion"
    )
    
    return Document(page_content=content, metadata=metadata)

def process_topic_posts(topic_data: Dict[str, Any]) -> Document:
    """Modified to avoid duplicating faculty posts already captured individually."""
    posts = [
        p for p in topic_data["posts"] 
        if p.get("user_title") != "Course_faculty"  # Exclude faculty posts already captured
    ]
    
    if not posts:
        return None
        
    weights = [calculate_post_weight(p) for p in posts]
    summary = generate_weighted_summary(posts, weights)
    
    metadata = {
        "source": topic_data["source_file"],
        "topic_id": str(topic_data["topic_id"]),
        "topic": topic_data["topic"],
        "topic_slug": topic_data["topic_slug"],
        "post_count": len(posts),
        "faculty_post_count": sum(1 for p in topic_data["posts"] if p.get("user_title") == "Course_faculty"),
        "total_weight": sum(weights),
        "document_type": "discourse_topic",
        "created_at": min(p.get("created_at", "") for p in topic_data["posts"]),
        "updated_at": max(p.get("updated_at", "") for p in topic_data["posts"])
    }
    
    return Document(page_content=summary, metadata=metadata)



def create_or_load_vector_store():
    """Create new or load existing Chroma vector store."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print("Loading existing vector store")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072
        )
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store")
        # Load documents
        discourse_docs = load_discourse_json_files(
            path= config.discourse_path,
            recursive=True
        )
        markdown_docs = load_markdown_files(
            path= config.markdown_path,
            recursive=True
        )

        # Combine documents
        docs = discourse_docs + markdown_docs 

        # Filter out empty documents BEFORE splitting
        filtered_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        if len(filtered_docs) != len(docs):
            print(f"Filtered out {len(docs) - len(filtered_docs)} empty documents")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(filtered_docs)
        
        # Debug: Verify splits
        print(f"\nAfter splitting: {len(splits)} chunks")
        for i, split in enumerate(splits[:3]):  # Print first 3 chunks
            print(f"\nChunk {i} preview:")
            print(f"Content length: {len(split.page_content)} chars")
            print(f"Content start: {split.page_content[:200]}...")
            print(f"Metadata: {split.metadata}")

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072
        )

        # Test embeddings
        try:
            test_embedding = embeddings.embed_query("Test embedding")
            print(f"\nEmbedding test successful. Vector length: {len(test_embedding)}")
        except Exception as e:
            print(f"\nEmbedding test failed: {str(e)}")
            raise

        # Create vector store
        try:
            vector_store = Chroma.from_documents(
                documents=splits,  # Use the split documents
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            print(f"\nSuccessfully created vector store with {len(splits)} chunks")
            print(f"Collection count: {vector_store._collection.count()}")
            return vector_store
        except Exception as e:
            print(f"\nFailed to create vector store: {str(e)}")
            raise


#for present deleting chroma db
def clear_chroma_persistence():
    if Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
    print(f"Cleared ChromaDB persistence at: {CHROMA_PERSIST_DIR}")

def load_markdown_with_metadata(file_path):
    """Simple markdown loader that extracts front matter metadata"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split front matter from content
    if content.startswith('---\n'):
        parts = content.split('---\n', 2)
        if len(parts) > 2:
            metadata = yaml.safe_load(parts[1])
            content = parts[2]
        else:
            metadata = {}
            content = parts[1]
    else:
        metadata = {}
    
    return content, metadata




def load_markdown_files(path: str, recursive: bool = True, exclude_dirs: List[str] = None) -> List[Document]:
    # Your modified processing loop

    if exclude_dirs is None:
        exclude_dirs = []
    
    documents = []
    md_extensions = ['.md', '.markdown', '.mdown']
    
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in md_extensions):
                file_path = os.path.join(root, file)
                
                # Load markdown and extract metadata
                content, metadata = load_markdown_with_metadata(file_path)
                
                # Create document (adjust based on your loader)
                loader = UnstructuredMarkdownLoader(file_path,preserve_metadata=True)
                docs = loader.load()
                
                # Update metadata for each doc
                for doc in docs:
                    doc.metadata.update({
                        **metadata,
                        "source_path": file_path,
                        "file_name": file,
                        "directory": root
                    })
                
                documents.extend(docs)
                
        
        if not recursive:
            break
    
    return documents


vector_store = create_or_load_vector_store()
print("Vector store created or loaded and persisted successfully")


