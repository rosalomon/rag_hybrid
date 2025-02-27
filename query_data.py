import argparse
from database_manager import DatabaseManager
from search_engine import SearchEngine
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
SYSTEM_PROMPT = """Du är en hjälpsam assistent som svarar på frågor om spelet Monopol baserat på given kontext.

VIKTIGT:
1. Ge ETT tydligt svar baserat ENDAST på informationen i kontexten
2. Om informationen inte finns i kontexten, svara endast: "Jag hittar ingen information om detta i kontexten"
3. Undvik att:
   - Gissa eller fylla i med egen kunskap
   - Ge flera alternativa svar
   - Svara på frågor som inte ställts
4. Citera relevant text från kontexten när möjligt."""

CHAT_PROMPT = """
Kontext:
{context}

Historik:
{history}

Användare: {question}
Assistent:"""

def format_source(metadata):
    """Formaterar källhänvisningen baserat på dokumenttyp"""
    source = metadata.get('source', 'Unknown')
    
    if metadata.get('type') == 'structured_data':
        sheet = metadata.get('sheet', '')
        row_start = metadata.get('row_start', '')
        row_end = metadata.get('row_end', '')
        return f"- {source} (Sheet: {sheet}, Rader: {row_start}-{row_end})"
    else:
        # För PDF/text-dokument
        page = metadata.get('page', '?')
        # Konvertera page till int om det är en siffra
        if isinstance(page, (int, float)):
            return f"- {source} (sida {int(page)})"
        return f"- {source} (sida {page})"

def main():
    db_manager = DatabaseManager(CHROMA_PATH)
    embedding_function = get_embedding_function()
    search_engine = SearchEngine(
        CHAT_PROMPT, 
        embedding_function,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Spara chat historik
    chat_history = []
    
    print("\n=== 🤖 RAG Chat ===")
    print("Skriv 'exit' för att avsluta\n")
    
    while True:
        # Få input från användaren
        user_input = input("\nFråga: ")
        if user_input.lower() in ['exit', 'quit', 'avsluta']:
            break
            
        # Sök efter relevanta dokument
        data = db_manager.get_all_documents()
        results = search_engine.search(user_input, data["documents"], data["metadatas"])
        
        # Skapa kontext från relevanta dokument
        context_text = "\n\n---\n\n".join([doc["page_content"] for doc, _score in results])
        
        # Formatera chat historik
        history_text = "\n".join([
            f"Användare: {q}\nAssistent: {a}" 
            for q, a in chat_history
        ])
        
        # Generera svar
        response_text = search_engine.generate_chat_response(
            user_input, 
            context_text,
            history_text
        )
        
        # Uppdatera historik
        chat_history.append((user_input, response_text))
        
        # Visa källor - Uppdatera för att hantera (doc, score) tupler
        sources = [format_source(doc[0]['metadata']) for doc in results]  # Notera doc[0] för att få dokumentet från tupeln
        print("\n📚 KÄLLOR:")
        print("\n".join(sorted(set(sources))))
        
        # Visa topp chunks
        print("\n🔍 TOPP 3 CHUNKS:")
        print("\nBM25 chunks:")
        for i, (doc, score) in enumerate(results[:3], 1):
            print(f"\n{i}. Score: {score:.3f}")
            print("-" * 50)
            print(doc["page_content"])
            print("-" * 50)
        
        if len(results) > 3:
            print("\nSimilarity chunks:")
            for i, (doc, score) in enumerate(results[3:6], 1):
                print(f"\n{i}. Score: {score:.3f}")
                print("-" * 50)
                print(doc["page_content"])
                print("-" * 50)
        # Visa svar
        print("\n=== ✨ SVAR ===")
        print(response_text)

if __name__ == "__main__":
    main()
