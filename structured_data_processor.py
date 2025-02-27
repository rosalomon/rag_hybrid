import pandas as pd
from langchain.schema.document import Document
from typing import List, Optional
import numpy as np
import importlib.util
import re

class StructuredDataProcessor:
    def __init__(self):
        # Kontrollera om nödvändiga paket är installerade
        self._check_dependencies()

    def _check_dependencies(self):
        """Kontrollerar om nödvändiga paket är installerade"""
        missing_packages = []
        
        # Kontrollera openpyxl
        if importlib.util.find_spec("openpyxl") is None:
            missing_packages.append("openpyxl")
        
        if missing_packages:
            print("\n⚠️ Saknade paket för att hantera strukturerad data:")
            print("Kör följande kommando för att installera:")
            print(f"pip install {' '.join(missing_packages)}")
            print("\n")

    def load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Laddar en Excel-fil och returnerar en DataFrame"""
        if importlib.util.find_spec("openpyxl") is None:
            print(f"⚠️ Hoppar över Excel-fil {file_path} - openpyxl är inte installerat")
            return pd.DataFrame()
            
        try:
            # Försök först läsa alla sheets
            if sheet_name is None:
                excel_file = pd.ExcelFile(file_path)
                all_sheets = []
                
                print(f"   Hittade {len(excel_file.sheet_names)} sheets:")
                for sheet in excel_file.sheet_names:
                    print(f"   - {sheet}")
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    if not df.empty:
                        # Lägg till sheet-namn som kolumn
                        df['Sheet'] = sheet
                        all_sheets.append(df)
                
                if all_sheets:
                    return pd.concat(all_sheets, ignore_index=True)
                return pd.DataFrame()
            
            # Om specifikt sheet anges
            return pd.read_excel(file_path, sheet_name=sheet_name)
        
        except Exception as e:
            print(f"❌ Fel vid inläsning av Excel-fil {file_path}: {e}")
            return pd.DataFrame()

    def load_csv(self, file_path: str, encoding: str = 'utf-8', sep: str = ',') -> pd.DataFrame:
        """Laddar en CSV-fil och returnerar en DataFrame"""
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=sep)
        except Exception as e:
            print(f"Fel vid inläsning av CSV-fil: {e}")
            return pd.DataFrame()

    def dataframe_to_documents(self, df: pd.DataFrame, source: str, chunk_size: int = 5) -> List[Document]:
        """Konverterar en DataFrame till en lista av Document-objekt med fokus på data"""
        documents = []
        
        # Gruppera data per sheet om det finns
        if 'Sheet' in df.columns:
            sheet_groups = df.groupby('Sheet')
        else:
            sheet_groups = [('main', df)]

        for sheet_name, sheet_df in sheet_groups:
            # Identifiera värdekolumner (numeriska kolumner)
            value_columns = sheet_df.select_dtypes(include=['float64', 'int64', 'object']).columns
            
            # Försök konvertera object-kolumner till numeriska
            numeric_columns = []
            for col in value_columns:
                if sheet_df[col].dtype == 'object':
                    # Testa första icke-null värdet
                    sample = sheet_df[col].dropna().iloc[0] if not sheet_df[col].dropna().empty else None
                    try:
                        if sample is not None:
                            pd.to_numeric(sample)
                            numeric_columns.append(col)
                    except:
                        continue
                else:
                    numeric_columns.append(col)
            
            value_columns = numeric_columns
            
            # Identifiera datumkolumner - var mer selektiv
            date_columns = []
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY eller MM/DD/YYYY
                r'\d{4}',              # Endast årtal
                r'\d{4}[Qq]\d{1}',     # År och kvartal (t.ex. 2019Q1 eller 2019q1)
            ]
            
            for col in sheet_df.columns:
                if sheet_df[col].dtype == 'datetime64[ns]':
                    date_columns.append(col)
                elif sheet_df[col].dtype == 'object':
                    # Testa första icke-null värdet mot datummönster
                    sample = sheet_df[col].dropna().iloc[0] if not sheet_df[col].dropna().empty else None
                    if sample is not None and isinstance(sample, str):
                        sample = sample.strip()
                        if any(re.match(pattern, sample) for pattern in date_patterns):
                            date_columns.append(col)
            
            # Identifiera beskrivande kolumner
            text_columns = [col for col in sheet_df.columns 
                           if col not in value_columns and 
                           col not in date_columns and 
                           col != 'Sheet' and 
                           not col.startswith('Unnamed:')]
            
            # Processa varje chunk av rader
            for start_idx in range(0, len(sheet_df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(sheet_df))
                chunk_df = sheet_df.iloc[start_idx:end_idx]
                
                content_parts = []
                
                # Lägg till diagram/tabell-beskrivning om den finns
                if text_columns:
                    for col in text_columns:
                        descriptions = chunk_df[col].dropna().unique()
                        if descriptions.size > 0:
                            content_parts.append(f"Beskrivning: {', '.join(str(d) for d in descriptions)}")
                
                # Lägg till data rad för rad
                for idx, row in chunk_df.iterrows():
                    row_parts = []
                    
                    # Lägg till datum om det finns
                    for date_col in date_columns:
                        if pd.notna(row[date_col]):
                            date_val = str(row[date_col]).strip()
                            row_parts.append(f"Datum ({date_col}): {date_val}")
                    
                    # Lägg till värden
                    for val_col in value_columns:
                        if pd.notna(row[val_col]):
                            try:
                                value = pd.to_numeric(row[val_col])
                                formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,d}"
                                row_parts.append(f"Värde ({val_col}): {formatted_value}")
                            except:
                                # Om värdet inte kan konverteras till numeriskt, lägg till som text
                                row_parts.append(f"Text ({val_col}): {row[val_col]}")
                    
                    if row_parts:
                        content_parts.append(" | ".join(row_parts))
                
                # Skapa metadata
                metadata = {
                    "source": source,
                    "type": "structured_data",
                    "sheet": sheet_name,
                    "row_start": start_idx + 1,
                    "row_end": end_idx,
                    "description": ", ".join(str(d) for d in descriptions) if text_columns and descriptions.size > 0 else "",
                    "value_columns": ", ".join(str(col) for col in value_columns),
                    "date_columns": ", ".join(str(col) for col in date_columns),
                    "id": f"{source}:{sheet_name}:{start_idx+1}-{end_idx}"
                }

                # Skapa Document-objekt endast om vi har meningsfullt innehåll
                if content_parts:
                    doc = Document(
                        page_content="\n".join(content_parts),
                        metadata=metadata
                    )
                    documents.append(doc)

        return documents

    def _format_chunk_as_text(self, df: pd.DataFrame) -> str:
        """Formaterar en DataFrame-chunk som läsbar text"""
        text_parts = []
        
        # Lägg till kolumnnamn
        text_parts.append("Kolumner:")
        for col in df.columns:
            text_parts.append(f"- {col}")
        
        # Lägg till data
        text_parts.append("\nData:")
        for idx, row in df.iterrows():
            text_parts.append(f"\nRad {idx+1}:")
            for col in df.columns:
                value = row[col]
                if pd.notna(value):  # Skippa NaN/None värden
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,d}"
                    else:
                        formatted_value = str(value)
                    text_parts.append(f"  {col}: {formatted_value}")
        
        return "\n".join(text_parts) 