import os
import json
import re
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def summarize_post(post_text: str, model="gpt-4o"):
    llm = ChatOpenAI(model=model, temperature=0.5)

    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes forum posts."),
        HumanMessage(content=f"Summarize the following post:\n\n{post_text}")
    ]

    response = llm.invoke(messages)
    return response.content  # AIMessage.content contains the summary


def clean_cooked_html(html_content):
    """
    Clean Discourse HTML content while preserving:
    - All text content
    - Code blocks
    - Links (as references)
    - Images (as captions with metadata)
    - Basic formatting (paragraphs, line breaks)
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Process images with metadata preservation
    for img in soup.find_all('img'):
        img.replace_with(" ")
    
    # Process links while preserving href
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if href:
            a.insert_after(" ")
            a.unwrap()
    
    # Process code blocks
    for pre in soup.find_all('pre'):
        code = pre.get_text()
        pre.replace_with(f"\n```\n{code}\n```\n")
    
    # Remove unnecessary divs/classes but keep content
    for div in soup.find_all('div'):
        div.unwrap()
    
    # Process lists
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.insert_before("- ")
            li.unwrap()
        ul.unwrap()
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li')):
            li.insert_before(f"{i+1}. ")
            li.unwrap()
        ol.unwrap()

    
    
    # Clean up line breaks and whitespace
    text = soup.get_text()
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Collapse multiple newlines
    text = text.strip()
    
    return text


def extract_mentions(html_content):
    """Extract mentioned usernames from post HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    mentions = set()
    for mention in soup.find_all('a', class_='mention'):
        href = mention.get('href', '')
        if href.startswith('/u/'):
            mentions.add(href[3:].lower())
    return mentions

def is_low_trust_user(user_data):
    """Check if user meets deletion criteria"""
    return (user_data.get('moderator') == False and 
            user_data.get('trust_level', 1) in (0, 1) and 
            user_data.get('primary_group_name') == 'ds-students')


def build_user_profiles(topic_data):
    """Create a username->profile mapping for mention checking"""
    user_profiles = {}
    for post in topic_data.get('post_stream', {}).get('posts', []):
        username = post.get('username', '').lower()
        if username and username not in user_profiles:
            user_profiles[username] = {
                'moderator': post.get('moderator', False),
                'trust_level': post.get('trust_level', 1),
                'primary_group_name': post.get('primary_group_name', '')
            }
    return user_profiles



def should_keep_post(post, topic_data, all_posts, user_profiles):
    """Determine if a post should be kept based on criteria"""
    # Always keep posts from allowed authors/faculty


    mentions = extract_mentions(post.get('cooked', ''))
    low_trust_mentions = False
    for username in mentions:
        if username in user_profiles and is_low_trust_user(user_profiles[username]):
            low_trust_mentions = True
            break

    if low_trust_mentions and post.get('score', 0) <= 60:
        return False

    allowed_authors = {'carlton', 'vanshbordia', 'jivraj', 's.anand'}
    author_username = post.get('username', '').lower()
    author_name = post.get('name', '').lower()

    if post['post_number'] == 1:
        author_username = post.get('username', '').lower()
        if author_username in allowed_authors:
            return False
    
    if (author_username in allowed_authors or 
        author_name in allowed_authors or 
        post.get('primary_group_name') == 'faculty'):
        return True
    
    soup = BeautifulSoup(post['cooked'], 'html.parser')

    if soup.find_all('img'):
        if post['score'] <= 60:
            return False


    # Check if post is from a low-trust user
    if is_low_trust_user(post):
        # Check if it has any replies from non-low-trust users
        has_quality_replies = False
        for reply in all_posts:
            if (reply.get('reply_to_post_number') == post['post_number'] and
                not is_low_trust_user(reply)):
                has_quality_replies = True
                break
        
        if not has_quality_replies:
            return False
        else:
            return True
    
    # Check unanswered directed questions
    if post.get('reply_count', 0) == 0:
        mentions = extract_mentions(post.get('cooked', ''))
        if mentions:
            return False
        
    if (post.get('moderator') == False and 
        post.get('trust_level') <= 1 and
        topic_data.get('vote_count', 0) == 0):
        return False
    
    return True
    
    

def filter_posts_in_topic(topic_data):
    """Filter posts within a single topic"""
    if 'post_stream' not in topic_data or 'posts' not in topic_data['post_stream']:
        return topic_data
    
    all_posts = topic_data['post_stream']['posts']
    user_profiles = build_user_profiles(topic_data)

    filtered_posts = [
        post for post in all_posts
        if should_keep_post(post, topic_data, all_posts, user_profiles)
    ]

    
    # Update topic metadata
    topic_data['post_stream']['posts'] = filtered_posts
    topic_data['posts_count'] = len(filtered_posts)
    
    # Recalculate reply counts
    reply_counts = {}
    for post in filtered_posts:
        reply_to = post.get('reply_to_post_number')
        if reply_to:
            reply_counts[reply_to] = reply_counts.get(reply_to, 0) + 1
    
    for post in filtered_posts:
        post['reply_count'] = reply_counts.get(post['post_number'], 0)
    
    topic_data['reply_count'] = sum(post.get('reply_count', 0) for post in filtered_posts)

    
    n = 0
    for post in filtered_posts:
        n+=1
        if 'cooked' in post:
            cleaned_text = clean_cooked_html(post['cooked'])
            post['cleaned_text'] = cleaned_text
            #post['cleaned_text'] = summarize_post(cleaned_text)
            #post['cleaned_text'] = summarize_post(post['cleaned_text'])
            print(f"happeneing {n}")
            
    
    return topic_data
    

def process_directory(input_dir, output_dir):
    """Process all JSON files recursively"""
    os.makedirs(output_dir, exist_ok=True)

    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                input_path = os.path.join(root, filename)
                
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        topic_data = json.load(f)
                    
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)

                    output_path = os.path.join(output_subdir, filename)

                    # Only write if file doesn't already exist
                    if not os.path.exists(output_path): 
                        filtered_topic = filter_posts_in_topic(topic_data)
                        
                        if filtered_topic['posts_count'] > 0:
                            os.makedirs(output_subdir, exist_ok=True)

                            output_path = os.path.join(output_subdir, filename)
                            
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(filtered_topic, f, indent=2)
                
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    #input_directory = r"C:\Users\yasme\Downloads\data\discourse_json"
    #output_directory = r"C:\Users\yasme\Downloads\data\filter_json"

    input_directory = r"C:\Users\yasme\Downloads\data\filter_json"
    output_directory = r"C:\Users\yasme\Downloads\data\cleaned_html"
    
    print(f"Processing files from {input_directory}...")
    process_directory(input_directory, output_directory)
    print(f"Filtered files saved to {output_directory}")