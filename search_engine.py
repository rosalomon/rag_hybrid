import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
import httpx
from typing import List, Dict, Tuple

class SearchEngine:
    def __init__(self, prompt_template, embedding_function=None, system_prompt=None):
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        self.embedding_function = embedding_function
        self.system_prompt = system_prompt

    def expand_query(self, query: str) -> str:
        """Expandera query med synonymer för bättre BM25 matchning"""
        try:
            # Använd språkmodellen för att generera relaterade termer
            prompt = f"Generate 2-3 synonyms or related terms for the query: {query}"
            expanded_terms = self.model.invoke(prompt).strip().split('\n')
            
            # Kombinera original query med expanderade termer
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            return expanded_query
        except:
            return query

    def search(self, query: str, documents: list[str], metadatas: list[dict], top_k_each=6) -> List[Tuple[Dict, float]]:
        results = []
        
        # BM25 sökning
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = word_tokenize(query.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Ta top-k från BM25
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_each]
        bm25_results = [({"page_content": documents[i], "metadata": metadatas[i]}, bm25_scores[i]) 
                        for i in bm25_indices if bm25_scores[i] > 0]
        results.extend(bm25_results)
        
        # Semantic similarity sökning
        if self.embedding_function:
            query_embedding = self.embedding_function.embed_query(query)
            doc_embeddings = [
                self.embedding_function.embed_documents([doc])[0] 
                for doc in documents
            ]
            
            similarity_scores = [
                np.dot(query_embedding, doc_embedding) 
                for doc_embedding in doc_embeddings
            ]
            
            # Ta top-k från similarity
            sim_indices = np.argsort(similarity_scores)[::-1][:top_k_each]
            sim_results = [({"page_content": documents[i], "metadata": metadatas[i]}, similarity_scores[i]) 
                          for i in sim_indices if similarity_scores[i] > 0]
            results.extend(sim_results)
        
        return results

    def generate_answer(self, query: str, context: str):
        # Uppdatera system prompt för att hantera strukturerad data
        structured_data_prompt = """
        Om svaret innehåller numerisk data eller tabelldata, formatera det tydligt.
        För numeriska värden:
        - Använd tusentalsavgränsare
        - Avrunda decimaler till två decimaler när lämpligt
        - Ange måttenheter när tillgängligt
        
        För tabelldata:
        - Presentera data i en läsbar struktur
        - Inkludera relevanta kolumnnamn
        - Ange källraden när möjligt
        """
        
        prompt = self.prompt_template.format(
            context=context,
            question=query,
            structured_data_instructions=structured_data_prompt
        )
        
        try:
            model = OpenAI(
                base_url="http://127.0.0.1:1234/v1",
                api_key="not-needed",
                temperature=0.2,
            
                top_p=0.95,
                model="meta-llama-3.1-8b-instruct"
            )
            response = model.invoke(prompt)
            return response.strip().replace("<|im_start|>", "").replace("<|im_end|>", "")
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_chat_response(self, query: str, context: str, history: str):
        # Formatera prompt som vanlig text istället för chat
        full_prompt = f"""
{self.system_prompt}

Kontext:
{context}

Historik:
{history}

Användare: {query}
Assistent:"""
        
        try:
            model = OpenAI(
                base_url="http://127.0.0.1:1234/v1",
                api_key="not-needed",
                temperature=0.4,
                model="meta-llama-3.1-8b-instruct",
                http_client=httpx.Client(timeout=30.0)
            )
            
            # Använd vanlig completion istället för chat completion
            response = model.invoke(full_prompt)
            return response.strip().replace("<|im_start|>", "").replace("<|im_end|>", "")
        except Exception as e:
            return f"Error: {str(e)}" 