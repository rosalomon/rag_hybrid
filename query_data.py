import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from get_embedding_function import get_embedding_function
from rank_bm25 import BM25Okapi
import numpy as np

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    print("\n=== 🔎 Sökning påbörjad ===")
    print(f"Fråga: {query_text}\n")
    
    # Hämta alla dokument från databasen för BM25-sökning.
    print("1. 📚 Hämtar dokument från databasen...")
    data = db.get()
    documents = data["documents"]  # Textinnehållet
    metadatas = data["metadatas"]  # Metadata för varje dokument
    print(f"   ✓ Hittade {len(documents)} dokument\n")

    # Bygg BM25-index baserat på dokumentinnehållet
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query_text.split()
    scores = bm25.get_scores(query_tokens)

    # Hämta de 12 bästa resultaten.
    top_k = 12
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [({"page_content": documents[i], "metadata": metadatas[i]}, scores[i]) 
               for i in top_indices if scores[i] > 0]
    print(f"2. 🔍 BM25-sökning klar")
    print(f"   ✓ Hittade {len(results)} relevanta dokument\n")

    context_text = "\n\n---\n\n".join(
        [doc["page_content"] for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("3. 🤖 Genererar svar från AI-modell...")
    
    try:
        import httpx
        model = OpenAI(
            base_url="http://127.0.0.1:1234/v1",  # LM Studio
            api_key="not-needed",
            temperature=0.7,
            model="meta-llama-3.1-8b-instruct",
            http_client=httpx.Client(timeout=30.0)
        )
        response = model.invoke(prompt)
        response_text = response.strip().replace("<|im_start|>", "").replace("<|im_end|>", "")
        if not response_text:
            response_text = "Error: Received empty response from LLM"
        print("   ✓ Svar mottaget\n")
    except Exception as e:
        print(f"\n❌ Fel vid anslutning till LM Studio: {str(e)}")
        print("Kontrollera att LM Studio körs och att modellen är laddad")
        return "Error: Could not connect to LM Studio"

    sources = [f"- {doc['metadata'].get('source', 'Unknown')} (sida {doc['metadata'].get('page', '?'):.0f})" 
              for doc, _score in results]
    
    print("\n=== ✨ RESULTAT ===")
    print("\n📝 SVAR:")
    print(response_text)
    print("\n📚 KÄLLOR:")
    print("\n".join(sources))
    print("\n================")


if __name__ == "__main__":
    main()
