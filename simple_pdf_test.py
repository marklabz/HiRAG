#!/usr/bin/env python3
"""
Simple PDF text extraction test without LLM dependencies
"""

import sys
sys.path.append('eval')

from pathlib import Path
from eval.advanced_document_processor import AdvancedDocumentProcessor


def simple_pdf_test():
    """Test PDF processing without LLM requirements"""
    
    print("=== Simple PDF Processing Test ===\n")
    
    pdf_dir = Path("lenderDocs/FlagstarBank/Flagstar_PDF")
    processor = AdvancedDocumentProcessor()
    
    # Test with a smaller PDF first
    test_pdf = "5010.pdf"
    pdf_path = pdf_dir / test_pdf
    
    if not pdf_path.exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        return
    
    print(f"Processing: {test_pdf}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
    result = processor.process_single_pdf(pdf_path)
    
    if result:
        text, metadata = result
        print(f"✅ Success!")
        print(f"   Extraction method: {metadata.extraction_method}")
        print(f"   Characters: {metadata.character_count:,}")
        print(f"   Processing time: {metadata.processing_time:.2f}s")
        
        # Show extracted content structure
        lines = text.split('\n')
        print(f"   Total lines: {len(lines)}")
        
        print(f"\n📄 Content preview (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            if line.strip():
                print(f"   {i+1:2d}: {line.strip()[:80]}")
        
        print(f"\n📊 Content analysis:")
        words = text.split()
        print(f"   Word count: {len(words):,}")
        print(f"   Average words per line: {len(words)/len(lines):.1f}")
        
        # Look for key financial/lending terms
        key_terms = ['mortgage', 'loan', 'credit', 'insurance', 'financing', 'property', 'borrower']
        found_terms = []
        for term in key_terms:
            if term.lower() in text.lower():
                count = text.lower().count(term.lower())
                found_terms.append(f"{term}: {count}")
        
        if found_terms:
            print(f"   Key lending terms found: {', '.join(found_terms[:5])}")
        
    else:
        print("❌ Failed to extract text")


if __name__ == "__main__":
    simple_pdf_test()
