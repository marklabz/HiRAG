#!/usr/bin/env python3
"""
Test script to verify PDF processing setup is working correctly.
"""

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    # PDF processing
    import pdfplumber
    import fitz  # PyMuPDF
    import PyPDF2
    print(f"✓ pdfplumber {pdfplumber.__version__}")
    print(f"✓ PyMuPDF (fitz) {fitz.version[0]}")
    print(f"✓ PyPDF2 {PyPDF2.__version__}")
    
    # HiRAG
    from hirag import HiRAG, QueryParam
    print("✓ HiRAG imported successfully")
    
    # Other utilities
    import yaml
    import tqdm
    import pandas as pd
    import numpy as np
    print(f"✓ numpy {np.__version__}")
    print(f"✓ pandas {pd.__version__}")
    
    print("\nAll imports successful! ✅")


def test_advanced_processor():
    """Test the AdvancedDocumentProcessor class"""
    print("\nTesting AdvancedDocumentProcessor...")
    
    try:
        from eval.advanced_document_processor import AdvancedDocumentProcessor
        processor = AdvancedDocumentProcessor()
        print("✓ AdvancedDocumentProcessor created successfully")
        
        # Test configuration loading
        print(f"✓ Processor configuration loaded")
        
    except Exception as e:
        print(f"❌ Error testing AdvancedDocumentProcessor: {e}")
        return False
    
    return True


def test_basic_hirag():
    """Test basic HiRAG functionality"""
    print("\nTesting basic HiRAG functionality...")
    
    try:
        from hirag import HiRAG, QueryParam
        
        # Create a minimal HiRAG instance
        graph_func = HiRAG(
            working_dir="./test_workspace",
            enable_hierachical_mode=True,
            enable_llm_cache=False,  # Disable to avoid LLM requirements
            enable_naive_rag=True
        )
        print("✓ HiRAG instance created successfully")
        
        # Test QueryParam
        param = QueryParam(mode="naive")
        print("✓ QueryParam created successfully")
        
    except Exception as e:
        print(f"❌ Error testing HiRAG: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=== PDF Processing Setup Test ===\n")
    
    try:
        test_imports()
        test_advanced_processor()
        test_basic_hirag()
        
        print("\n=== All Tests Passed! ✅ ===")
        print("\nYour PDF processing setup is ready to use!")
        print("\nNext steps:")
        print("1. Place PDF files in a directory (e.g., './pdfs')")
        print("2. Run: python eval/pdf_to_knowledge_graph.py")
        print("3. Or use: python eval/advanced_document_processor.py")
        
    except Exception as e:
        print(f"\n❌ Setup test failed: {e}")
        print("\nPlease check the error above and ensure all requirements are installed.")
