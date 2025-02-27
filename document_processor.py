import nltk
from transformers import AutoTokenizer
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from nltk.tokenize import PunktSentenceTokenizer
from structured_data_processor import StructuredDataProcessor
import os
from typing import List
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader

class DocumentProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.structured_processor = StructuredDataProcessor()
        
        # Använder en flerspråkig modell som stödjer både svenska och engelska
        self.tokenizer = AutoTokenizer.from_pretrained("KBLab/sentence-bert-swedish-cased")
        
        # Ladda ner punkt-tokenizer data
        nltk.download('punkt', quiet=True)
        
        # Skapa en generell sentence splitter
        self.sentence_splitter = PunktSentenceTokenizer()

    def load_documents(self):
        documents = []
        
        print("\n=== 📂 Laddar dokument ===")
        
        # Ladda PDF-filer
        pdf_files = [f for f in os.listdir(self.data_path) if f.endswith('.pdf')]
        if pdf_files:
            print(f"\n📄 Hittade {len(pdf_files)} PDF-filer")
            pdf_loader = PyPDFDirectoryLoader(self.data_path)
            try:
                pdf_docs = pdf_loader.load()
                print(f"   Laddade {len(pdf_docs)} PDF-dokument")
                
                pdf_chunks = []
                for doc in pdf_docs:
                    try:
                        chunks = self._split_text_document(doc)
                        pdf_chunks.extend(chunks)
                    except Exception as e:
                        print(f"   ⚠️ Fel vid chunkning av PDF: {str(e)}")
                
                print(f"   Skapade {len(pdf_chunks)} chunks från PDF-filer")
                documents.extend(pdf_chunks)
                
            except Exception as e:
                print(f"❌ Fel vid laddning av PDF-filer: {str(e)}")
        
        # Ladda Excel-filer
        excel_files = [f for f in os.listdir(self.data_path) if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            print(f"\n📊 Hittade {len(excel_files)} Excel-filer")
            for file in excel_files:
                file_path = os.path.join(self.data_path, file)
                print(f"\nBearbetar: {file}")
                
                try:
                    # Använd UnstructuredExcelLoader utan elements mode för att få ren text
                    excel_loader = UnstructuredExcelLoader(file_path)  # Ta bort mode="elements"
                    excel_docs = excel_loader.load()
                    
                    # Lägg till metadata
                    for doc in excel_docs:
                        doc.metadata.update({
                            "source": file,
                            "type": "structured_data"
                        })
                    
                    print(f"   Laddade {len(excel_docs)} element från Excel-fil")
                    documents.extend(excel_docs)
                    
                except Exception as e:
                    print(f"❌ Fel vid laddning av Excel-fil {file}: {str(e)}")
        
        # Ladda CSV-filer
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if csv_files:
            print(f"\n📑 Hittade {len(csv_files)} CSV-filer")
            for file in csv_files:
                file_path = os.path.join(self.data_path, file)
                print(f"\nBearbetar: {file}")
                
                try:
                    # Använd LangChains CSVLoader
                    csv_loader = CSVLoader(file_path)
                    csv_docs = csv_loader.load()
                    
                    # Lägg till metadata
                    for doc in csv_docs:
                        doc.metadata.update({
                            "source": file,
                            "type": "structured_data"
                        })
                    
                    print(f"   Laddade {len(csv_docs)} rader från CSV-fil")
                    documents.extend(csv_docs)
                    
                except Exception as e:
                    print(f"❌ Fel vid laddning av CSV-fil {file}: {str(e)}")
        
        print(f"\n✅ Totalt laddade dokument: {len(documents)}")
        
        return documents

    def _split_text_document(self, doc: Document) -> List[Document]:
        """Dela upp ett textdokument i chunks"""
        chunks = []
        
        # Kontrollera att input är ett Document-objekt
        if not isinstance(doc, Document):
            print(f"⚠️ Varning: Ogiltigt dokument format i _split_text_document: {type(doc)}")
            return chunks
        
        try:
            sentences = self.sentence_splitter.tokenize(doc.page_content)
            
            current_chunk = []
            current_length = 0
            last_sentence = None
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > 250:
                    if current_length >= 150:
                        # Lägg till chunk om den är tillräckligt stor
                        chunk_text = ' '.join(current_chunk)
                        # Skapa nytt Document-objekt med kopia av metadata
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=dict(doc.metadata)  # Skapa en kopia av metadata
                        ))
                        
                        # Starta ny chunk med överlappning
                        current_chunk = [last_sentence, sentence] if last_sentence else [sentence]
                        current_length = len(last_sentence) + sentence_length if last_sentence else sentence_length
                    else:
                        # Om chunken är för liten, fortsätt lägga till
                        current_chunk.append(sentence)
                        current_length += sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                
                last_sentence = sentence
            
            # Lägg till sista chunken om den inte är tom
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata)  # Skapa en kopia av metadata
                ))
            
            # Verifiera att alla chunks är Document-objekt
            chunks = [chunk for chunk in chunks if isinstance(chunk, Document)]
            
        except Exception as e:
            print(f"⚠️ Fel vid chunkning av dokument: {str(e)}")
            return []
        
        return chunks

    def inspect_chunks(self, chunks: list[Document], num_samples=5):
        """Visa detaljerad information om chunks för inspektion"""
        print(f"\n=== 📝 Chunk-inspektion (visar {num_samples} exempel) ===")
        print(f"Totalt antal chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks[:num_samples]):
            print(f"\nChunk {i+1}:")
            print("Metadata:", chunk.metadata)
            print("Längd (tecken):", len(chunk.page_content))
            print("Antal meningar:", len(self.sentence_splitter.tokenize(chunk.page_content)))
            print("Innehåll:")
            print("-" * 50)
            # Visa meningar på separata rader för tydlighet
            sentences = self.sentence_splitter.tokenize(chunk.page_content)
            for j, sentence in enumerate(sentences, 1):
                print(f"{j}. {sentence}")
            print("-" * 50)
        
        # Visa statistik
        lengths = [len(chunk.page_content) for chunk in chunks]
        sentence_counts = [len(self.sentence_splitter.tokenize(chunk.page_content)) 
                         for chunk in chunks]
        
        print("\n📊 Statistik för alla chunks:")
        print(f"Genomsnittlig längd: {sum(lengths)/len(lengths):.1f} tecken")
        print(f"Min längd: {min(lengths)} tecken")
        print(f"Max längd: {max(lengths)} tecken")
        print(f"Genomsnittligt antal meningar: {sum(sentence_counts)/len(sentence_counts):.1f}")
        print("=" * 50 + "\n") 