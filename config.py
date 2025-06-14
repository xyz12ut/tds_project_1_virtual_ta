# config.py
from dotenv import load_dotenv
import os

load_dotenv("variables.env")  # Load from file in current directory

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


discourse_path = r"C:\Users\yasme\Downloads\data\summariezed_filtered_json"
markdown_path = r"C:\Users\yasme\Downloads\data\tds_pages_md"