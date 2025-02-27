import os
import shutil
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import chromadb
import time
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import Dict, Any

COLLECTION_NAME = "documents"

class DatabaseManager:
    def __init__(self, chroma_path):
        self.chroma_path = chroma_path
        self._initialize_db()

    def _initialize_db(self):
        self.db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=get_embedding_function(),
            collection_name=COLLECTION_NAME
        )

    def clear_database(self):
        # Stäng och rensa den befintliga databasen
        if hasattr(self, 'db'):
            try:
                self.db.delete_collection()
            except:
                pass
            del self.db
        
        # Vänta en kort stund
        time.sleep(1)
        
        # Ta bort hela chroma-mappen och dess innehåll
        if os.path.exists(self.chroma_path):
            try:
                shutil.rmtree(self.chroma_path, ignore_errors=True)
            except Exception as e:
                print(f"Varning: Kunde inte ta bort alla filer: {e}")
        
        # Vänta igen för att säkerställa att allt är borta
        time.sleep(1)
        
        # Återskapa databasen
        self._initialize_db()

    def add_documents(self, chunks: list[Document]):
        print(f"\n=== 🔍 Debug Information ===")
        print(f"Inkommande chunks: {len(chunks)}")
        
        def debug_metadata(metadata: Dict[str, Any], prefix: str = ""):
            """Hjälpfunktion för att debugga metadata"""
            for key, value in metadata.items():
                print(f"{prefix}{key}: ({type(value).__name__}) = {value}")
        
        def safe_filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            """Säker filtrering av metadata som behåller dictionary-strukturen"""
            filtered = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    filtered[key] = value
                elif isinstance(value, list):
                    # Konvertera lista till sträng
                    filtered[key] = ", ".join(str(v) for v in value)
                else:
                    # För andra typer, konvertera till sträng
                    filtered[key] = str(value)
            return filtered
        
        # Verifiera chunks och filtrera metadata
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, Document):
                print(f"Chunk {i}: Inte ett Document-objekt, typ: {type(chunk)}")
                continue
            
            try:
                print(f"\nBearbetar chunk {i}:")
                print("Original metadata:")
                debug_metadata(chunk.metadata, "  ")
                
                # Använd vår egen säkra filtrering istället för filter_complex_metadata
                filtered_metadata = safe_filter_metadata(chunk.metadata)
                
                print("\nEfter säker filtrering:")
                debug_metadata(filtered_metadata, "  ")
                
                # Lägg till ID om det inte finns
                if "id" not in filtered_metadata:
                    if filtered_metadata.get("type") == "structured_data":
                        source = filtered_metadata.get("source", "unknown")
                        sheet = filtered_metadata.get("sheet", "main")
                        row_start = filtered_metadata.get("row_start", 0)
                        row_end = filtered_metadata.get("row_end", 0)
                        filtered_metadata["id"] = f"{source}:{sheet}:{row_start}-{row_end}"
                    else:
                        source = filtered_metadata.get("source", "unknown")
                        page = filtered_metadata.get("page", 0)
                        chunk_index = getattr(self, '_chunk_index', 0)
                        filtered_metadata["id"] = f"{source}:{page}:{chunk_index}"
                        self._chunk_index = chunk_index + 1
                
                print("\nSlutlig metadata:")
                debug_metadata(filtered_metadata, "  ")
                
                # Skapa ny Document med filtrerad metadata
                valid_chunks.append(Document(
                    page_content=chunk.page_content,
                    metadata=filtered_metadata
                ))
                
            except Exception as e:
                print(f"⚠️ Fel vid bearbetning av chunk {i}: {str(e)}")
                print(f"Metadata som orsakade felet:")
                debug_metadata(chunk.metadata, "  ")
                continue
        
        if not valid_chunks:
            print("❌ Inga giltiga chunks att lägga till")
            return
        
        print(f"\n✅ Filtrerade {len(chunks)} chunks till {len(valid_chunks)} giltiga chunks")
        
        try:
            # Lägg till dokumenten
            self.db.add_documents(valid_chunks)
            print(f"✅ Lade till {len(valid_chunks)} chunks i databasen")
        except Exception as e:
            print(f"❌ Fel vid tillägg till databasen: {str(e)}")
            # Visa exempel på metadata som orsakade felet
            if valid_chunks:
                print("\nExempel på problematisk metadata:")
                debug_metadata(valid_chunks[0].metadata, "  ")

    def _calculate_chunk_ids(self, chunks):
        """Beräknar unika ID:n för chunks"""
        for chunk in chunks:
            if chunk.metadata.get("type") == "structured_data":
                # För strukturerad data, använd befintligt ID om det finns
                if "id" not in chunk.metadata:
                    source = chunk.metadata.get("source", "unknown")
                    sheet = chunk.metadata.get("sheet", "main")
                    row_start = chunk.metadata.get("row_start", 0)
                    row_end = chunk.metadata.get("row_end", 0)
                    chunk.metadata["id"] = f"{source}:{sheet}:{row_start}-{row_end}"
            else:
                # Hantering av PDF/text-dokument som tidigare
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)
                current_chunk_index = getattr(self, '_chunk_index', 0)
                chunk.metadata["id"] = f"{source}:{page}:{current_chunk_index}"
                self._chunk_index = current_chunk_index + 1

        return chunks

    def get_all_documents(self):
        return self.db.get() 