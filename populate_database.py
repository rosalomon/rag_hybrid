import argparse
from document_processor import DocumentProcessor
from database_manager import DatabaseManager

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--inspect", action="store_true", help="Inspect chunks before adding to database.")
    args = parser.parse_args()

    db_manager = DatabaseManager(CHROMA_PATH)
    if args.reset:
        print("✨ Rensar databasen")
        db_manager.clear_database()

    doc_processor = DocumentProcessor(DATA_PATH)
    
    # Ladda och chunka dokument
    documents = doc_processor.load_documents()
    
    if args.inspect:
        doc_processor.inspect_chunks(documents)
    
    # Lägg till dokumenten i databasen
    db_manager.add_documents(documents)

if __name__ == "__main__":
    main()
