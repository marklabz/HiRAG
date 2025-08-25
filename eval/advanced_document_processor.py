#!/usr/bin/env python3
"""
Advanced document processing for HiRAG knowledge graphs.
Supports multiple document formats and advanced text preprocessing.
"""

import os
import json
import logging
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
from dataclasses import dataclass
from datetime import datetime

# PDF processing
import PyPDF2
import fitz  # PyMuPDF
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# OCR support
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Text processing
import re
from tqdm import tqdm

from hirag import HiRAG, QueryParam


@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    filename: str
    file_path: str
    file_size: int
    extraction_method: str
    processing_time: float
    character_count: int
    page_count: Optional[int] = None
    extraction_errors: List[str] = None


class AdvancedDocumentProcessor:
    """Advanced document processor with multiple extraction methods and preprocessing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.processed_docs: List[DocumentMetadata] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text using PyPDF2"""
        text = ""
        errors = []
        page_count = 0
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    except Exception as e:
                        errors.append(f"Page {i+1}: {str(e)}")
                        
        except Exception as e:
            errors.append(f"File reading error: {str(e)}")
            
        return text.strip(), errors
    
    def extract_text_pymupdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text using PyMuPDF (fitz)"""
        text = ""
        errors = []
        page_count = 0
        
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            for i, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
                except Exception as e:
                    errors.append(f"Page {i+1}: {str(e)}")
            
            doc.close()
            
        except Exception as e:
            errors.append(f"File reading error: {str(e)}")
            
        return text.strip(), errors
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text using pdfplumber (better for tables)"""
        if not HAS_PDFPLUMBER:
            return "", ["pdfplumber not available"]
            
        text = ""
        errors = []
        page_count = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {i+1} ---\n{page_text}\n"
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            text += f"\n--- Table {j+1} on Page {i+1} ---\n"
                            for row in table:
                                text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            
                    except Exception as e:
                        errors.append(f"Page {i+1}: {str(e)}")
                        
        except Exception as e:
            errors.append(f"File reading error: {str(e)}")
            
        return text.strip(), errors
    
    def extract_text_with_ocr(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text using OCR for scanned PDFs"""
        if not HAS_OCR:
            return "", ["OCR libraries not available"]
            
        text = ""
        errors = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for i, page in enumerate(doc):
                try:
                    # Convert page to image
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # OCR the image
                    image = Image.open(io.BytesIO(img_data))
                    page_text = pytesseract.image_to_string(image)
                    
                    text += f"\n--- Page {i+1} (OCR) ---\n{page_text}\n"
                    
                except Exception as e:
                    errors.append(f"OCR Page {i+1}: {str(e)}")
            
            doc.close()
            
        except Exception as e:
            errors.append(f"OCR processing error: {str(e)}")
            
        return text.strip(), errors
    
    def preprocess_text(self, text: str, filename: str) -> str:
        """Advanced text preprocessing"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Add document header with metadata
        header = f"Document: {filename}\nExtracted on: {datetime.now().isoformat()}\n\n"
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'\f', '\n', text)  # Replace form feeds
        
        return header + text.strip()
    
    def process_single_pdf(self, pdf_path: Path) -> Optional[Tuple[str, DocumentMetadata]]:
        """Process a single PDF file with multiple extraction methods"""
        start_time = datetime.now()
        
        # Try different extraction methods in order of preference
        # pdfplumber is prioritized for its excellent table and layout handling
        extraction_methods = [
            ("pdfplumber", self.extract_text_pdfplumber),
            ("pymupdf", self.extract_text_pymupdf),
            ("pypdf2", self.extract_text_pypdf2),
        ]
        
        # Add OCR as fallback if available
        if HAS_OCR:
            extraction_methods.append(("ocr", self.extract_text_with_ocr))
        
        best_text = ""
        best_method = ""
        all_errors = []
        
        for method_name, method_func in extraction_methods:
            try:
                text, errors = method_func(str(pdf_path))
                all_errors.extend([f"{method_name}: {e}" for e in errors])
                
                # Use the first method that produces substantial text
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
                    best_method = method_name
                    
                # If we got good text, stop trying other methods
                if len(text.strip()) > 100:  # Arbitrary threshold
                    break
                    
            except Exception as e:
                all_errors.append(f"{method_name}: {str(e)}")
                continue
        
        if not best_text.strip():
            self.logger.warning(f"No text extracted from {pdf_path.name}")
            return None
        
        # Preprocess the text
        processed_text = self.preprocess_text(best_text, pdf_path.name)
        
        # Create metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        metadata = DocumentMetadata(
            filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size=pdf_path.stat().st_size,
            extraction_method=best_method,
            processing_time=processing_time,
            character_count=len(processed_text),
            extraction_errors=all_errors if all_errors else None
        )
        
        self.processed_docs.append(metadata)
        return processed_text, metadata
    
    def process_document_directory(self, directory: str) -> List[str]:
        """Process all supported documents in a directory"""
        directory = Path(directory)
        
        # Find all PDF files
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return []
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        documents = []
        
        # Process files with progress bar
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.process_single_pdf(pdf_file)
            if result:
                text, metadata = result
                documents.append(text)
                self.logger.info(f"Processed {pdf_file.name}: {metadata.character_count} chars")
            else:
                self.logger.error(f"Failed to process {pdf_file.name}")
        
        return documents
    
    def save_processing_report(self, output_path: str):
        """Save a detailed processing report"""
        report = {
            "processing_summary": {
                "total_documents": len(self.processed_docs),
                "successful_extractions": len([d for d in self.processed_docs if d.character_count > 0]),
                "total_characters": sum(d.character_count for d in self.processed_docs),
                "average_processing_time": sum(d.processing_time for d in self.processed_docs) / len(self.processed_docs) if self.processed_docs else 0,
            },
            "document_details": [
                {
                    "filename": doc.filename,
                    "file_size_mb": round(doc.file_size / (1024 * 1024), 2),
                    "extraction_method": doc.extraction_method,
                    "processing_time_seconds": round(doc.processing_time, 2),
                    "character_count": doc.character_count,
                    "has_errors": bool(doc.extraction_errors),
                    "errors": doc.extraction_errors
                }
                for doc in self.processed_docs
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Processing report saved to {output_path}")


def main():
    """Example usage of the advanced document processor"""
    processor = AdvancedDocumentProcessor("config.yaml")
    
    # Process documents
    documents = processor.process_document_directory("./pdfs")
    
    if documents:
        # Save processing report
        processor.save_processing_report("./processing_report.json")
        
        # Create HiRAG instance and build knowledge graph
        graph_func = HiRAG(
            working_dir="./hirag_workspace",
            enable_hierachical_mode=True,
            enable_llm_cache=True,
            enable_naive_rag=True
        )
        
        # Insert documents in batches
        batch_size = 3
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            graph_func.insert(batch)
        
        print("Knowledge graph construction complete!")
        
        # Test query
        response = graph_func.query(
            "What are the main topics discussed in these documents?",
            param=QueryParam(mode="hi")
        )
        print(f"Test query response: {response}")
    
    else:
        print("No documents were successfully processed")


if __name__ == "__main__":
    main()
