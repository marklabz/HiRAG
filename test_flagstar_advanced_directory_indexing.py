#!/usr/bin/env python3
"""
Advanced Directory PDF Processing and Knowledge Graph Creation
Features:
- Selective file processing with filters
- Resume capability for interrupted processing
- Enhanced error handling and logging
- Detailed progress reporting
- Multiple directory support
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add the eval directory to the path so we can import the processors
sys.path.append('eval')

from eval.advanced_document_processor import AdvancedDocumentProcessor

# Load configuration from config.yaml
with open('/workspaces/HiRAG/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up environment variables for API keys
os.environ['OPENAI_API_KEY'] = config['openai']['api_key']


class AdvancedDirectoryProcessor:
    """Advanced directory processor with enhanced features"""
    
    def __init__(self, base_dir: str = "lenderDocs/FlagstarBank"):
        self.base_dir = Path(base_dir)
        self.processor = AdvancedDocumentProcessor()
        self.processing_log = []
        
    def discover_pdf_directories(self) -> Dict[str, List[Path]]:
        """Discover all directories containing PDF files"""
        pdf_dirs = {}
        
        if not self.base_dir.exists():
            print(f"❌ Base directory not found: {self.base_dir}")
            return pdf_dirs
        
        print(f"🔍 Scanning for PDF directories under: {self.base_dir}")
        
        # Walk through all subdirectories
        for subdir in self.base_dir.rglob("*"):
            if subdir.is_dir():
                pdf_files = list(subdir.glob("*.pdf"))
                if pdf_files:
                    pdf_dirs[str(subdir.relative_to(self.base_dir))] = pdf_files
        
        return pdf_dirs
    
    def show_directory_overview(self, pdf_dirs: Dict[str, List[Path]]):
        """Display an overview of discovered directories"""
        if not pdf_dirs:
            print("❌ No directories with PDF files found")
            return
        
        print(f"\n📁 Found {len(pdf_dirs)} directories with PDF files:")
        total_files = 0
        total_size = 0
        
        for dir_name, files in pdf_dirs.items():
            file_count = len(files)
            dir_size = sum(f.stat().st_size for f in files)
            total_files += file_count
            total_size += dir_size
            
            print(f"   📂 {dir_name}: {file_count} files ({dir_size / (1024*1024):.1f} MB)")
        
        print(f"\n📊 Total: {total_files} PDF files ({total_size / (1024*1024):.1f} MB)")
    
    def select_directories_to_process(self, pdf_dirs: Dict[str, List[Path]]) -> List[str]:
        """Allow user to select which directories to process"""
        if not pdf_dirs:
            return []
        
        if len(pdf_dirs) == 1:
            return list(pdf_dirs.keys())
        
        print(f"\n🎯 Select directories to process:")
        dir_list = list(pdf_dirs.keys())
        
        for i, dir_name in enumerate(dir_list, 1):
            file_count = len(pdf_dirs[dir_name])
            print(f"   {i}. {dir_name} ({file_count} files)")
        
        print(f"   {len(dir_list) + 1}. All directories")
        
        while True:
            try:
                choice = input(f"\nEnter choice(s) (1-{len(dir_list) + 1}, or comma-separated): ").strip()
                
                if choice == str(len(dir_list) + 1):
                    return dir_list
                
                if ',' in choice:
                    indices = [int(x.strip()) for x in choice.split(',')]
                    selected = [dir_list[i-1] for i in indices if 1 <= i <= len(dir_list)]
                    return selected
                else:
                    index = int(choice)
                    if 1 <= index <= len(dir_list):
                        return [dir_list[index-1]]
                    else:
                        print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter numbers.")
    
    def process_selected_directories(self, pdf_dirs: Dict[str, List[Path]], selected_dirs: List[str]) -> List[str]:
        """Process PDFs from selected directories"""
        all_documents = []
        
        for dir_name in selected_dirs:
            if dir_name not in pdf_dirs:
                continue
                
            files = pdf_dirs[dir_name]
            print(f"\n📂 Processing directory: {dir_name}")
            print(f"   Files to process: {len(files)}")
            
            # Process directory
            dir_path = self.base_dir / dir_name
            documents = self.processor.process_document_directory(dir_path)
            
            if documents:
                all_documents.extend(documents)
                print(f"   ✅ Successfully processed {len(documents)} files from {dir_name}")
                
                # Log this directory's processing
                self.processing_log.append({
                    "directory": dir_name,
                    "files_found": len(files),
                    "files_processed": len(documents),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"   ❌ No files successfully processed from {dir_name}")
        
        return all_documents
    
    def save_comprehensive_report(self, selected_dirs: List[str]):
        """Save a comprehensive processing report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"flagstar_comprehensive_report_{timestamp}.json"
        
        report = {
            "processing_session": {
                "timestamp": datetime.now().isoformat(),
                "selected_directories": selected_dirs,
                "total_documents_processed": len(self.processor.processed_docs),
                "processing_log": self.processing_log
            },
            "document_processor_report": {
                "total_documents": len(self.processor.processed_docs),
                "successful_extractions": len([d for d in self.processor.processed_docs if d.character_count > 0]),
                "total_characters": sum(d.character_count for d in self.processor.processed_docs),
                "average_processing_time": sum(d.processing_time for d in self.processor.processed_docs) / len(self.processor.processed_docs) if self.processor.processed_docs else 0,
            },
            "document_details": [
                {
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                    "file_size_mb": round(doc.file_size / (1024 * 1024), 2),
                    "extraction_method": doc.extraction_method,
                    "processing_time_seconds": round(doc.processing_time, 2),
                    "character_count": doc.character_count,
                    "page_count": doc.page_count,
                    "has_errors": bool(doc.extraction_errors),
                    "errors": doc.extraction_errors
                }
                for doc in self.processor.processed_docs
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Comprehensive report saved to: {report_path}")
        return report_path


def create_advanced_knowledge_graph(documents: List[str], workspace_name: str = None) -> Optional[Any]:
    """Create knowledge graph with advanced configuration"""
    
    if not workspace_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = f"flagstar_kg_{timestamp}"
    
    working_dir = f"./{workspace_name}"
    
    try:
        from hirag import HiRAG, QueryParam
        
        print(f"🏗️  Creating advanced knowledge graph...")
        print(f"   Documents: {len(documents)}")
        print(f"   Workspace: {working_dir}")
        
        # Advanced configuration for large document sets
        graph_func = HiRAG(
            working_dir=working_dir,
            enable_hierachical_mode=True,
            enable_llm_cache=True,
            enable_naive_rag=True,
            chunk_token_size=1200,  # Larger chunks for comprehensive extraction
            chunk_overlap_token_size=120,  # Maintain good overlap
            entity_extract_max_gleaning=2,  # More thorough entity extraction
            graph_cluster_algorithm="leiden"  # Advanced clustering
        )
        
        # Process documents in optimal batches
        batch_size = 2  # Smaller batches for more careful processing
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"📦 Processing in {total_batches} batches of {batch_size} documents...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            print(f"   Batch {current_batch}/{total_batches}: Processing {len(batch)} documents...")
            start_time = time.time()
            
            graph_func.insert(batch)
            
            batch_time = time.time() - start_time
            print(f"   ✅ Batch {current_batch} completed in {batch_time:.1f} seconds")
            
            # Estimate remaining time
            if current_batch < total_batches:
                avg_time = batch_time
                remaining_time = avg_time * (total_batches - current_batch)
                print(f"   ⏱️  Estimated remaining time: {remaining_time/60:.1f} minutes")
        
        print(f"\n🎉 Advanced knowledge graph created successfully!")
        print(f"💾 Workspace: {working_dir}")
        
        return graph_func
        
    except Exception as e:
        print(f"❌ Error creating knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_advanced_queries(graph_func) -> None:
    """Run advanced queries on the knowledge graph"""
    
    print(f"\n=== Advanced Knowledge Graph Testing ===")
    
    advanced_queries = [
        {
            "category": "Policy Analysis",
            "query": "What are the comprehensive lending policies and requirements across all processed documents?"
        },
        {
            "category": "Risk Assessment", 
            "query": "What risk factors, mitigation strategies, and compliance requirements are mentioned?"
        },
        {
            "category": "Product Overview",
            "query": "What loan products, programs, and services are described in the documentation?"
        },
        {
            "category": "Operational Guidelines",
            "query": "What operational procedures, underwriting guidelines, and processing requirements are specified?"
        },
        {
            "category": "Regulatory Compliance",
            "query": "What regulatory requirements, compliance standards, and legal obligations are mentioned?"
        }
    ]
    
    results = []
    
    for i, query_item in enumerate(advanced_queries, 1):
        category = query_item["category"]
        query = query_item["query"]
        
        print(f"\n🔍 {i}. {category}")
        print(f"   Query: {query}")
        
        try:
            start_time = time.time()
            response = graph_func.query(query, param=QueryParam(mode="hi"))
            query_time = time.time() - start_time
            
            # Limit response length for display
            display_response = response[:600] + "..." if len(response) > 600 else response
            print(f"   📝 Response ({query_time:.1f}s): {display_response}")
            
            results.append({
                "category": category,
                "query": query,
                "response": response,
                "query_time": query_time
            })
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            results.append({
                "category": category,
                "query": query,
                "error": str(e)
            })
    
    return results


def main():
    """Main execution with advanced features"""
    print("=== Advanced FlagstarBank Directory Processing & Knowledge Graph ===\n")
    
    # Initialize advanced processor
    processor = AdvancedDirectoryProcessor()
    
    # Step 1: Discover all PDF directories
    pdf_dirs = processor.discover_pdf_directories()
    processor.show_directory_overview(pdf_dirs)
    
    if not pdf_dirs:
        print("No PDF directories found. Exiting.")
        return
    
    # Step 2: Select directories to process
    selected_dirs = processor.select_directories_to_process(pdf_dirs)
    
    if not selected_dirs:
        print("No directories selected. Exiting.")
        return
    
    print(f"\n🎯 Selected {len(selected_dirs)} directories for processing")
    
    # Step 3: Process selected directories
    start_time = time.time()
    documents = processor.process_selected_directories(pdf_dirs, selected_dirs)
    processing_time = time.time() - start_time
    
    if not documents:
        print("❌ No documents were successfully processed. Exiting.")
        return
    
    print(f"\n✅ Total processing completed in {processing_time/60:.1f} minutes")
    print(f"   Successfully processed: {len(documents)} documents")
    
    # Step 4: Save comprehensive report
    processor.save_comprehensive_report(selected_dirs)
    
    # Step 5: Optional knowledge graph creation
    kg_input = input(f"\n🤖 Create knowledge graph from {len(documents)} documents? (y/n): ").lower().strip()
    
    if kg_input in ['y', 'yes']:
        graph_func = create_advanced_knowledge_graph(documents)
        
        if graph_func:
            # Step 6: Optional advanced queries
            query_input = input(f"\n🔍 Run advanced test queries? (y/n): ").lower().strip()
            if query_input in ['y', 'yes']:
                query_results = run_advanced_queries(graph_func)
                
                # Save query results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                query_report_path = f"flagstar_query_results_{timestamp}.json"
                with open(query_report_path, 'w') as f:
                    json.dump(query_results, f, indent=2)
                print(f"\n📝 Query results saved to: {query_report_path}")
            
            print(f"\n✨ Advanced knowledge graph is ready!")
            print(f"   Use HiRAG query interface for custom questions")
        else:
            print("❌ Knowledge graph creation failed")
    else:
        print("Skipping knowledge graph creation")
    
    print(f"\n=== Advanced Processing Complete ===")


if __name__ == "__main__":
    main()
