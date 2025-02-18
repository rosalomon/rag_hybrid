import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

from get_embedding_function import get_embedding_function

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

    # Search the DB.
    print("🔍 Searching database...")
    results = db.similarity_search_with_score(query_text, k=5)
    print(f"✅ Found {len(results)} results")

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("📝 Generated prompt")
    print(prompt)

    print("🤖 Connecting to LM Studio...")
    try:
        import httpx
        model = OpenAI(
            base_url="http://127.0.0.1:1234/v1",  # LM Studio
            api_key="not-needed",
            temperature=0.7,
            model="qwen2.5-7b-instruct-1m",
            http_client=httpx.Client(timeout=30.0)
        )
        print("🚀 Sending request to LLM...")
        response = model.invoke(prompt)
        # Rensa bort eventuella system-tokens från svaret
        response_text = response.strip().replace("<|im_start|>", "").replace("<|im_end|>", "")
        if not response_text:
            response_text = "Error: Received empty response from LLM"
        print("✅ Received response")
    except Exception as e:
        print(f"❌ Error connecting to LM Studio: {str(e)}")
        print("Please check if LM Studio is running and the model is loaded")
        return "Error: Could not connect to LM Studio"

    sources = [doc.metadata.get("source", "Unknown") + f" (page {doc.metadata.get('page', '?')})" 
              for doc, _score in results]
    formatted_response = f"\nResponse: {response_text}\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
