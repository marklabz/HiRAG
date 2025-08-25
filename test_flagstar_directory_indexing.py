#!/usr/bin/env python3
"""
Directory PDF Processing and Knowledge Graph Creation with FlagstarBank Documents
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


def process_flagstar_directory():
    """Process entire FlagstarBank PDF directory"""
    
    # Path to the FlagstarBank PDFs directory
    pdf_dir = Path("lenderDocs/FlagstarBank/Flagstar_PDF")
    
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return None, None
    
    print(f"🔍 Scanning directory: {pdf_dir}")
    
    # Count PDFs in directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return None, None
    
    print(f"📁 Found {len(pdf_files)} PDF files to process")
    print(f"   Directory size: {sum(f.stat().st_size for f in pdf_files) / (1024*1024):.1f} MB")
    
    # Initialize the document processor
    processor = AdvancedDocumentProcessor()
    
    print(f"\n🚀 Starting directory processing...")
    print(f"   This may take several minutes depending on file sizes and complexity")
    
    # Process the entire directory
    documents = processor.process_document_directory(pdf_dir)
    
    if documents:
        print(f"\n✅ Successfully processed {len(documents)} documents!")
        
        # Generate processing report
        report_path = "flagstar_directory_processing_report.json"
        processor.save_processing_report(report_path)
        print(f"📊 Detailed processing report saved to: {report_path}")
        
        # Show summary statistics
        total_chars = sum(len(doc) for doc in documents)
        print(f"\n📈 Processing Summary:")
        print(f"   Total documents processed: {len(documents)}")
        print(f"   Total characters extracted: {total_chars:,}")
        print(f"   Average document size: {total_chars // len(documents):,} characters")
        
        successful_docs = len([d for d in processor.processed_docs if d.character_count > 0])
        print(f"   Successful extractions: {successful_docs}/{len(processor.processed_docs)}")
        
        if successful_docs < len(processor.processed_docs):
            failed_docs = len(processor.processed_docs) - successful_docs
            print(f"   ⚠️  Failed extractions: {failed_docs}")
        
        return documents, processor
    else:
        print(f"❌ No documents were successfully processed")
        return None, None


def build_knowledge_graph_from_directory(documents, processor):
    """Build a knowledge graph from all processed documents"""
    
    print(f"\n=== Building Knowledge Graph from Directory ===")
    
    try:
        from hirag import HiRAG, QueryParam
        
        # Create directory-specific workspace
        working_dir = "./flagstar_directory_workspace"
        
        print(f"🏗️  Initializing HiRAG knowledge graph...")
        print(f"   Workspace: {working_dir}")
        
        # Initialize HiRAG with settings optimized for larger document sets
        graph_func = HiRAG(
            working_dir=working_dir,
            enable_hierachical_mode=True,
            enable_llm_cache=True,
            enable_naive_rag=True,
            chunk_token_size=1000,  # Slightly larger chunks for comprehensive docs
            chunk_overlap_token_size=100
        )
        
        print(f"📝 Processing {len(documents)} documents into knowledge graph...")
        print(f"   This will involve LLM processing and may take 10-20 minutes...")
        
        # Process documents in batches to manage memory and API calls
        batch_size = 3  # Process 3 documents at a time
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            print(f"   📦 Processing batch {current_batch}/{total_batches} ({len(batch)} documents)")
            
            # Get the corresponding file names for this batch
            batch_files = []
            if processor and len(processor.processed_docs) >= i + len(batch):
                for j in range(len(batch)):
                    if i + j < len(processor.processed_docs):
                        batch_files.append(processor.processed_docs[i + j].filename)
            
            if batch_files:
                print(f"     Files: {', '.join(batch_files)}")
            
            graph_func.insert(batch)
            print(f"     ✅ Batch {current_batch} completed")
        
        print(f"\n🎉 Knowledge graph construction complete!")
        print(f"💾 All data saved in: {working_dir}")
        
        return graph_func
        
    except ImportError as e:
        print(f"❌ HiRAG import error: {e}")
        print("   Make sure HiRAG is properly installed")
        return None
    except Exception as e:
        print(f"❌ Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_knowledge_graph_queries(graph_func):
    """Test the knowledge graph with sample queries"""
    
    print(f"\n=== Testing Knowledge Graph Queries ===")
    
    test_queries = [
        "What are the main lending requirements and policies mentioned across all documents?",
        "What credit overlays and underwriting guidelines are specified?",
        "What are the key compliance requirements mentioned in the documents?",
        "What loan programs and products are described?",
        "What are the main risk factors and mitigation strategies discussed?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        try:
            response = graph_func.query(query, param=QueryParam(mode="hi"))
            # Truncate very long responses for readability
            display_response = response[:800] + "..." if len(response) > 800 else response
            print(f"📝 Response: {display_response}")
        except Exception as e:
            print(f"❌ Query failed: {e}")


def main():
    """Main execution function"""
    print("=== FlagstarBank Directory PDF Processing & Knowledge Graph Creation ===\n")
    
    # Step 1: Process entire directory
    documents, processor = process_flagstar_directory()
    
    if not documents:
        print("Exiting due to document processing failure.")
        return
    
    # Step 2: Build knowledge graph
    user_input = input(f"\n🤖 Build knowledge graph from {len(documents)} documents? This requires LLM access and may take 10-20 minutes (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        graph_func = build_knowledge_graph_from_directory(documents, processor)
        
        if graph_func:
            # Step 3: Test queries
            test_input = input(f"\n🔍 Test the knowledge graph with sample queries? (y/n): ").lower().strip()
            if test_input in ['y', 'yes']:
                test_knowledge_graph_queries(graph_func)
            
            print(f"\n✨ Complete! Your knowledge graph is ready for queries.")
            print(f"   Use the HiRAG query interface to ask questions about your documents.")
        else:
            print("❌ Knowledge graph creation failed.")
    else:
        print("Skipping knowledge graph creation.")
    
    print(f"\n=== Directory Processing Complete ===")


if __name__ == "__main__":
    main()
