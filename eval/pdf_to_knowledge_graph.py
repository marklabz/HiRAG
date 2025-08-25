#!/usr/bin/env python3
"""
Complete pipeline for building HiRAG knowledge graphs from PDF documents.
This script demonstrates how to process multiple PDF files and create a knowledge graph from scratch.
"""

import os
import json
import PyPDF2
import fitz  # PyMuPDF - better for complex PDFs
from pathlib import Path
from typing import List, Dict, Any
import yaml

from hirag import HiRAG, QueryParam


def extract_text_from_pdf_pypdf2(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2 (simpler, good for basic PDFs)"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path} with PyPDF2: {e}")
    return text.strip()


def extract_text_from_pdf_pymupdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (better for complex layouts)"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path} with PyMuPDF: {e}")
    return text.strip()


def process_pdf_directory(pdf_directory: str, use_pymupdf: bool = True) -> List[str]:
    """
    Process all PDF files in a directory and extract text content.
    
    Args:
        pdf_directory: Path to directory containing PDF files
        use_pymupdf: Whether to use PyMuPDF (True) or PyPDF2 (False)
    
    Returns:
        List of extracted text content from all PDFs
    """
    pdf_directory = Path(pdf_directory)
    pdf_files = list(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    documents = []
    extract_func = extract_text_from_pdf_pymupdf if use_pymupdf else extract_text_from_pdf_pypdf2
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        text = extract_func(str(pdf_file))
        
        if text.strip():
            # Add document metadata as header
            doc_text = f"Document: {pdf_file.name}\n\n{text}"
            documents.append(doc_text)
            print(f"  Extracted {len(text)} characters")
        else:
            print(f"  Warning: No text extracted from {pdf_file.name}")
    
    return documents


def create_hirag_instance(working_dir: str, config_path: str = "config.yaml") -> HiRAG:
    """
    Create and configure HiRAG instance.
    
    Args:
        working_dir: Directory where HiRAG will store its data
        config_path: Path to configuration file
    
    Returns:
        Configured HiRAG instance
    """
    # Load configuration if available
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create working directory
    os.makedirs(working_dir, exist_ok=True)
    
    # Configure HiRAG with optimal settings for knowledge graph creation
    hirag_config = config.get('hirag', {})
    
    graph_func = HiRAG(
        working_dir=working_dir,
        enable_llm_cache=hirag_config.get('enable_llm_cache', True),
        enable_hierachical_mode=hirag_config.get('enable_hierachical_mode', True),  # Enable hierarchical extraction
        embedding_batch_num=hirag_config.get('embedding_batch_num', 6),
        embedding_func_max_async=hirag_config.get('embedding_func_max_async', 8),
        enable_naive_rag=hirag_config.get('enable_naive_rag', True),
        # Chunking parameters for better entity extraction
        chunk_token_size=hirag_config.get('chunk_token_size', 1200),
        chunk_overlap_token_size=hirag_config.get('chunk_overlap_token_size', 100),
        # Entity extraction parameters
        entity_extract_max_gleaning=hirag_config.get('entity_extract_max_gleaning', 1),
        # Graph clustering
        graph_cluster_algorithm=hirag_config.get('graph_cluster_algorithm', 'leiden'),
    )
    
    return graph_func


def build_knowledge_graph_from_pdfs(
    pdf_directory: str,
    working_dir: str,
    config_path: str = "config.yaml",
    batch_size: int = 5
) -> HiRAG:
    """
    Complete pipeline to build knowledge graph from PDF documents.
    
    Args:
        pdf_directory: Directory containing PDF files
        working_dir: Directory for HiRAG storage
        config_path: Configuration file path
        batch_size: Number of documents to process in each batch
    
    Returns:
        HiRAG instance with built knowledge graph
    """
    print("=== PDF to Knowledge Graph Pipeline ===")
    
    # Step 1: Extract text from PDFs
    print("\n1. Extracting text from PDF documents...")
    documents = process_pdf_directory(pdf_directory)
    
    if not documents:
        raise ValueError("No documents were successfully processed")
    
    print(f"Successfully extracted text from {len(documents)} documents")
    
    # Step 2: Create HiRAG instance
    print("\n2. Initializing HiRAG...")
    graph_func = create_hirag_instance(working_dir, config_path)
    
    # Step 3: Build knowledge graph in batches
    print("\n3. Building knowledge graph...")
    print("This process will:")
    print("   - Chunk documents into manageable pieces")
    print("   - Extract entities and relationships using LLM")
    print("   - Create hierarchical knowledge structure")
    print("   - Build community reports")
    print("   - Generate embeddings for retrieval")
    
    # Process documents in batches to manage memory and API costs
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        # Insert documents into knowledge graph
        graph_func.insert(batch)
        
        print(f"Batch {batch_num} completed successfully")
    
    print("\n=== Knowledge Graph Construction Complete ===")
    print(f"Graph data stored in: {working_dir}")
    
    return graph_func


def query_knowledge_graph(graph_func: HiRAG, queries: List[str]) -> Dict[str, str]:
    """
    Test the knowledge graph with sample queries.
    
    Args:
        graph_func: Built HiRAG instance
        queries: List of test queries
    
    Returns:
        Dictionary mapping queries to responses
    """
    print("\n=== Testing Knowledge Graph ===")
    results = {}
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Use hierarchical mode for best results
        response = graph_func.query(query, param=QueryParam(mode="hi"))
        results[query] = response
        
        print(f"Response: {response}")
    
    return results


def main():
    """Main execution function"""
    # Configuration
    PDF_DIRECTORY = "./pdfs"  # Directory containing your PDF files
    WORKING_DIR = "./hirag_workspace"  # Where HiRAG stores its data
    CONFIG_PATH = "config.yaml"  # Configuration file
    
    # Sample test queries
    TEST_QUERIES = [
        "What are the main topics discussed in these documents?",
        "Who are the key people mentioned?",
        "What organizations are referenced?",
        "What are the main events described?",
        "Summarize the key relationships between entities"
    ]
    
    try:
        # Build knowledge graph from PDFs
        graph_func = build_knowledge_graph_from_pdfs(
            pdf_directory=PDF_DIRECTORY,
            working_dir=WORKING_DIR,
            config_path=CONFIG_PATH,
            batch_size=3  # Adjust based on your documents and API limits
        )
        
        # Test the knowledge graph
        results = query_knowledge_graph(graph_func, TEST_QUERIES)
        
        # Save results
        results_file = os.path.join(WORKING_DIR, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
