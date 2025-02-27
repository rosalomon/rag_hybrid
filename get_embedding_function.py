from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

def get_embedding_function():
    # Använder en modell tränad för dot product similarity
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )
    return embeddings
