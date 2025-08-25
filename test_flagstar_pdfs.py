#!/usr/bin/env python3
"""
Test PDF processing with actual FlagstarBank PDF documents
"""

import os
import sys
import yaml
from pathlib import Path

# Add the eval directory to the path so we can import the processors
sys.path.append('eval')

from eval.advanced_document_processor import AdvancedDocumentProcessor

# Load configuration from config.yaml
with open('/workspaces/HiRAG/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up environment variables for API keys
os.environ['OPENAI_API_KEY'] = config['openai']['api_key']


def test_flagstar_pdfs():
    """Test PDF processing with FlagstarBank documents"""
    
    # Path to the FlagstarBank PDFs
    pdf_dir = Path("lenderDocs/FlagstarBank/Flagstar_PDF")
    
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return
    
    print(f"Testing PDF processing with FlagstarBank documents from: {pdf_dir}")
    
    # Select a couple of test PDFs
    test_pdfs = [
        "Flagstar_ConventionalLoanCreditOverlays.pdf",
        "Flagstar_UWConv.pdf",
        "5010.pdf"  # Try a numbered one too
    ]
    
    processor = AdvancedDocumentProcessor()
    
    for pdf_name in test_pdfs:
        pdf_path = pdf_dir / pdf_name
        
        if not pdf_path.exists():
            print(f"⚠️  PDF not found: {pdf_name}")
            continue
            
        print(f"\n=== Testing {pdf_name} ===")
        print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        try:
            result = processor.process_single_pdf(pdf_path)
            
            if result:
                text, metadata = result
                print(f"✅ Successfully processed!")
                print(f"   Method used: {metadata.extraction_method}")
                print(f"   Characters extracted: {metadata.character_count:,}")
                print(f"   Processing time: {metadata.processing_time:.2f} seconds")
                
                if metadata.extraction_errors:
                    print(f"   Warnings/Errors: {len(metadata.extraction_errors)}")
                    for error in metadata.extraction_errors[:3]:  # Show first 3 errors
                        print(f"     - {error}")
                
                # Show a sample of extracted text
                preview = text[:500].replace('\n', ' ').strip()
                print(f"   Text preview: {preview}...")
                
            else:
                print(f"❌ Failed to extract text from {pdf_name}")
                
        except Exception as e:
            print(f"❌ Error processing {pdf_name}: {e}")
    
    # Generate processing report
    if processor.processed_docs:
        report_path = "flagstar_processing_report.json"
        processor.save_processing_report(report_path)
        print(f"\n📊 Processing report saved to: {report_path}")


def test_pdf_to_knowledge_graph():
    """Test building a small knowledge graph from FlagstarBank PDFs"""
    
    print("\n=== Testing Knowledge Graph Creation ===")
    
    try:
        from hirag import HiRAG, QueryParam
        
        # Create a small test workspace
        working_dir = "./flagstar_test_workspace"
        
        # Initialize HiRAG
        graph_func = HiRAG(
            working_dir=working_dir,
            enable_hierachical_mode=True,
            enable_llm_cache=True,
            enable_naive_rag=True,
            chunk_token_size=800,  # Smaller chunks for testing
            chunk_overlap_token_size=50
        )
        
        # Process a couple of PDFs
        processor = AdvancedDocumentProcessor()
        pdf_dir = Path("lenderDocs/FlagstarBank/Flagstar_PDF")
        
        test_pdfs = [
            "Flagstar_ConventionalLoanCreditOverlays.pdf",
            "Flagstar_UWConv.pdf"
        ]
        
        documents = []
        for pdf_name in test_pdfs:
            pdf_path = pdf_dir / pdf_name
            if pdf_path.exists():
                result = processor.process_single_pdf(pdf_path)
                if result:
                    text, metadata = result
                    documents.append(text)
                    print(f"✅ Added {pdf_name} to knowledge graph input")
        
        if documents:
            print(f"\n🏗️  Building knowledge graph from {len(documents)} documents...")
            print("   This may take a few minutes as it involves LLM processing...")
            
            # Insert documents into the knowledge graph
            graph_func.insert(documents)
            
            print("✅ Knowledge graph built successfully!")
            
            # Test a query
            test_query = "What are the main lending requirements mentioned in these documents?"
            print(f"\n🔍 Testing query: {test_query}")
            
            response = graph_func.query(test_query, param=QueryParam(mode="hi"))
            print(f"📝 Response: {response}")
            
            print(f"\n💾 Knowledge graph data saved in: {working_dir}")
            
        else:
            print("❌ No documents were successfully processed for knowledge graph")
            
    except Exception as e:
        print(f"❌ Error in knowledge graph test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=== FlagstarBank PDF Processing Test ===\n")
    
    # Test 1: PDF processing
    test_flagstar_pdfs()
    
    # Test 2: Knowledge graph creation (optional, requires LLM)
    user_input = input("\n🤖 Do you want to test knowledge graph creation? This requires LLM access (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        test_pdf_to_knowledge_graph()
    else:
        print("Skipping knowledge graph test.")
    
    print("\n=== Test Complete ===")
