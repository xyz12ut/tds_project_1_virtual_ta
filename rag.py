from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
from typing import TypedDict, Optional, List
from langchain_core.prompts import ChatPromptTemplate
import json
import config
import io
import base64
from PIL import Image



# Initialize vector store first
from setup_vectorstore import vector_store

# Initialize components
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# State definition
class Link(TypedDict):
    url: str
    text: str

class State(TypedDict):
    question: str
    image: Optional[str]  # None if no image
    context: List[Document]  # List of LangChain Documents
    answer: str  # Updated answer (initially empty)
    links: List[Link]  # New field for sources

# Query expansion
expansion_prompt = PromptTemplate.from_template("""
As an expert search query expander, generate 3-5 variations of this query that would help retrieve the most relevant documents.
Focus on:
- Synonym substitution
- Adding related concepts
- Including both broad and specific versions

Original query: {query}

Respond ONLY with a bulleted list of expanded queries formatted exactly like this:
- Expanded query 1
- Expanded query 2
- Expanded query 3
""")

query_expander = (
    {"query": RunnablePassthrough()} 
    | expansion_prompt
    | llm
    | StrOutputParser()
    | (lambda text: [line[2:].strip() for line in text.split("\n") if line.strip()])
)



def process_image_question(image_url: str, question: str) -> str:
    # Decide the type of image input
    if image_url.startswith("file:///"):
        file_path = image_url[8:].replace('/', '\\')  # Windows path
        image_type = "local"
    elif image_url.startswith("file://"):
        file_path = image_url[7:]  # Unix path
        image_type = "local"
    elif image_url.startswith("http://") or image_url.startswith("https://"):
        image_type = "web"
    elif image_url.startswith("data:image"):
        image_type = "base64"
    else:
        try:
            # Try decoding directly — if it's a valid base64, accept it
            base64.b64decode(image_url)
            image_type = "base64_raw"
        except Exception:
            raise ValueError("Unsupported image URL format or invalid base64")

    if image_type == "web":
        image_payload = {"url": image_url}

    elif image_type == "base64":
        # Extract base64 content from data URI
        base64_image = image_url.split(";base64,")[-1]
        image_payload = {"url": f"data:image/jpeg;base64,{base64_image}"}

    elif image_type == "base64_raw":
        image_payload = {"url": f"data:image/jpeg;base64,{image_url}"}

    elif image_type == "local":
        try:
            with Image.open(file_path) as img:
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                image_payload = {"url": f"data:image/jpeg;base64,{base64_image}"}
        except Exception as e:
            raise ValueError(f"Could not process local image: {str(e)}")

    # Create and send message to the model
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": image_payload}
        ]
    )

    return chat.invoke([message]).content




def retrieve(state: State) -> dict:
    if state.get("image"):
        description = process_image_question(state["image"], state.get("question", "Describe this image"))
        retrieved_docs = vector_store.similarity_search(description)
    else:
        question = state["question"]
        expanded_queries = [question] + query_expander.invoke(question)
        
        # Get documents without immediate filtering
        
        all_docs = []
        for q in expanded_queries:
            all_docs.extend(vector_store.similarity_search(q, k=3))
        

        retrieved_docs = []
        seen_contents = set()
        
        for doc in all_docs:
            # Create a content signature to detect near-duplicates
            content_hash = hash(doc.page_content[:200])  # First 200 chars as signature
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                retrieved_docs.append(doc)
                
                # Stop when we have enough unique content
                '''
                if len(retrieved_docs) >= 5:
                    break
                '''

    return {"context": retrieved_docs}

def generate(state: State) -> dict:
    prompt_template = """Answer the question based only on the following context.
    If you don't know the answer, say you don't know. Be precise and stick to the facts.
    Also try to provide additional context and information to the answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    docs_content = "\n\n".join(f"Document {i+1}: {doc.page_content}" 
                             for i, doc in enumerate(state["context"]))
    
    # Format the prompt
    messages = prompt.invoke({
        "question": state["question"], 
        "context": docs_content
    })
    
    response = llm.invoke(messages)
    
    # Extract links - now using original_url correctly
    links = []
    seen_urls = set()
    
    for doc in state["context"]:
        url = doc.metadata.get("original_url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            snippet = doc.page_content[:100].strip()
            if len(doc.page_content) > 100:
                snippet += "..."
            links.append({
                "url": url,
                "text": snippet
            })
    
    return {
        "answer": response.content,
        "links": links
    }


def format_response(raw_response):
    # Extract the answer - handle both string and JSON string cases
    answer = raw_response.get("answer", "")
    
    # If answer appears to be a JSON string (like in first example), extract the actual answer
    if answer.startswith('```json') and '"answer":' in answer:
        try:
            json_start = answer.find('{')
            json_end = answer.rfind('}') + 1
            json_str = answer[json_start:json_end]
            answer_data = json.loads(json_str)
            answer = answer_data.get("answer", answer)
        except:
            pass  # If parsing fails, keep original answer
    
    # Process links - handle both formats
    formatted_links = []
    
    # Case 1: Links are in the raw_response's "links" key (second example)
    if "links" in raw_response and isinstance(raw_response["links"], list):
        for link in raw_response["links"]:
            if isinstance(link, dict):
                url = link.get("url", "")
                raw_text = link.get("text", "")
                
                # Clean up the text
                cleaned_text = clean_link_text(raw_text)
                
                if url and cleaned_text:
                    formatted_links.append({
                        "url": url,
                        "text": cleaned_text
                    })
    
    # Case 2: Links might be embedded in a JSON string answer (first example)
    elif '"links":' in answer:
        try:
            if answer.startswith('```json'):
                json_start = answer.find('{')
                json_end = answer.rfind('}') + 1
                json_str = answer[json_start:json_end]
                answer_data = json.loads(json_str)
            else:
                answer_data = json.loads(answer)
            
            for link in answer_data.get("links", []):
                url = link.get("url", "")
                raw_text = link.get("text", "")
                cleaned_text = clean_link_text(raw_text)
                
                if url and cleaned_text:
                    formatted_links.append({
                        "url": url,
                        "text": cleaned_text
                    })
        except:
            pass
    
    return {
        "answer": answer.replace('```json', '').replace('```', '').strip(),
        "links": formatted_links
    }

def clean_link_text(raw_text):
    """Helper function to clean up link text"""
    if not raw_text:
        return ""
    
    # Take first line if there are multiple lines
    cleaned_text = raw_text.split('\n')[0]
    
    # Remove metadata markers
    cleaned_text = cleaned_text.replace('title: "', '').replace('original_url: "', '')
    cleaned_text = cleaned_text.split('"')[0].strip()
    
    # Remove markdown code blocks if present
    cleaned_text = cleaned_text.replace('```', '')
    
    # Remove excessive whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Truncate if too long
    if len(cleaned_text) > 150:
        cleaned_text = cleaned_text[:147] + "..."
    
    return cleaned_text


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



if __name__ == "__main__":
    response = graph.invoke({"question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?" ,"image": "file:///C:/Users/yasme/Downloads/project-tds-virtual-ta-q1.webp" })
    #response = graph.invoke({"question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"})
    formatted_response = format_response(response)
    print(formatted_response)
    #print(response)
    print("success")