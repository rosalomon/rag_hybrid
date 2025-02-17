
from langchain_openai import OpenAIEmbeddings
import os

def get_embedding_function():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings
