import json
import os
import time
from openai import OpenAI
from collections import defaultdict
import re
import config



def extract_main_value(text):
    # First, check if the string contains quoted text
    quote_match = re.search(r'"(.*?)"', text)
    if quote_match:
        return quote_match.group(1).strip()
    
    single_quote_match = re.search(r"'(.*?)'", text)
    if single_quote_match:
        return single_quote_match.group(1).strip()
    
    # If no quotes, check for known prefixes
    match = re.search(r'(?:New topic name:|Existing topic name:|Classified post as:|New topic:|Existing topic:)\s*(.+)', text)
    if match:
        return match.group(1).strip()

    # If none of the patterns match, return the raw string
    return text.strip()




# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
def classify_post(text, thread_topics):
    """
    Classify a post into a topic or mark as irrelevant.
    Each post is classified independently, but we'll check against existing topics later.
    """
    prompt = f"""
    Analyze this forum post from a course discussion between students and faculty.
    Correctly classify this text as a topic.
    Current thread topics: {thread_topics if thread_topics else "No topics yet"}
    
    Instructions:
    1. Analyze the text very carefully and be very detailed
    2. If the post clearly belongs to an existing topic, return that exact topic name
    3. If the post is relevant but doesn't fit existing topics suggest a new topic 
    4. Be very specific and detailed
    5. If the post is irrelevant (social, informal, duplicates, or not important), return "irrelevant"
    6. Be specific about the issue, the problem , the post while classifying same topic posts together.
    
    Post content: "{text}"
    
    Return ONLY one of these:
    - Existing topic name (exact match)
    - New topic name (if relevant)
    - "irrelevant"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.4,
            max_tokens=70
        )
        classification = response.choices[0].message.content.strip()
        return classification
        
    except Exception as e:
        print(f"Error classifying post: {e}")
        return 'irrelevant'

def process_thread(posts):
    """
    Process a single thread to classify posts with topic validation
    """
    topic_counts = defaultdict(int)
    thread_topics = set()  # Reset for each new thread
    
    for post in posts:
        if 'cleaned_text' not in post or not post['cleaned_text'].strip():
            post['topic'] = 'irrelevant'
            continue
            
        text = post['cleaned_text']
        
        # Add delay to avoid rate limiting
        time.sleep(1)  # 1 second delay between API calls
        
        classification = classify_post(text, list(thread_topics))  # Pass current topics
        middle = extract_main_value(classification)
        classification = extract_main_value(middle)
        
        # Validate classification against existing topics
        if classification.lower() == 'irrelevant':
            post['topic'] = 'irrelevant'
        elif classification in thread_topics:
            post['topic'] = classification
            topic_counts[classification] += 1
        else:
            post['topic'] = classification
            thread_topics.add(classification)
            topic_counts[classification] += 1
        
        print(f"{post['topic']}")
    
    # Add thread's topics to metadata
    if posts:
        first_post = posts[0]
        if 'thread_metadata' not in first_post:
            first_post['thread_metadata'] = {}
        first_post['thread_metadata']['topics'] = list(thread_topics)
        first_post['thread_metadata']['topic_counts'] = dict(topic_counts)
    
    return posts

def process_json_file(filepath):
    """Process a single JSON file with better error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'post_stream' in data and 'posts' in data['post_stream']:
            posts = data['post_stream']['posts']
            print(f"Processing {len(posts)} posts in {filepath}")
            
            processed_posts = process_thread(posts)
            data['post_stream']['posts'] = processed_posts
            
            # Save the processed data back to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return processed_posts
        return []
    
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    return None

def process_directory(root_dir):
    """Process all JSON files in a directory recursively with progress tracking"""
    processed_files = 0
    error_files = 0
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                print(f"\nProcessing {filepath}...")
                
                result = process_json_file(filepath)
                if result is None:
                    error_files += 1
                else:
                    processed_files += 1
                
                print(f"Progress: {processed_files} files processed, {error_files} errors")
    
    print(f"\nProcessing complete! {processed_files} files processed, {error_files} errors encountered")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify forum posts into thread-specific topics')
    parser.add_argument('directory', help='Directory containing JSON files to process')
    args = parser.parse_args()
    
    process_directory(args.directory)