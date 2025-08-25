"""
RAG Performance Testing Framework
Tests different chunk sizes and overlap configurations
"""
import os
import time
import shutil
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Import our modules
from test_questions import test_cases
from evaluation_metrics import evaluate_retrieval_quality, format_results_summary

# Import RAG components
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import glob

load_dotenv()

class RAGPerformanceTester:
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        
    def load_documents(self) -> List[Document]:
        """Load all documents (same as buildDatabase.py)"""
        # Load markdown files
        md_loader = DirectoryLoader(self.data_path, glob="*.md")
        md_docs = md_loader.load()
        
        # Load PDF files using PyPDFLoader and combine pages
        pdf_docs = []
        pdf_files = glob.glob(os.path.join(self.data_path, "*.pdf"))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            # Combine all pages into one document to preserve context
            if pages:
                combined_text = "\n\n".join([page.page_content for page in pages])
                combined_doc = Document(
                    page_content=combined_text, 
                    metadata={"source": pdf_file}
                )
                pdf_docs.append(combined_doc)
        
        return md_docs + pdf_docs
    
    def build_database(self, chunk_size: int, overlap: int, chroma_path: str) -> float:
        """Build vector database with specified configuration"""
        start_time = time.time()
        
        # Remove existing database
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        
        # Load and split documents
        docs = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        
        # Build vector database
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        
        build_time = time.time() - start_time
        print(f"Built database with {len(chunks)} chunks in {build_time:.2f} seconds")
        return build_time
    
    def run_query_test(self, question: str, chroma_path: str) -> List[Tuple[Document, float]]:
        """Run a single query and return results"""
        embedding_func = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_func)
        
        results = db.similarity_search_with_relevance_scores(question, k=3)
        return results
    
    def test_configuration(self, config: Dict) -> Dict:
        """Test a single configuration"""
        chunk_size = config['chunk_size']
        overlap = config['overlap']
        config_name = f"chunk_{chunk_size}_overlap_{overlap}"
        chroma_path = f"test_chroma_{config_name}"
        
        print(f"\nTesting configuration: {config_name}")
        print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
        
        # Build database
        build_time = self.build_database(chunk_size, overlap, chroma_path)
        
        # Test all questions
        results = []
        for test_case in test_cases:
            print(f"  Testing: {test_case['question'][:50]}...")
            
            # Run query
            query_results = self.run_query_test(test_case['question'], chroma_path)
            
            # Evaluate results
            metrics = evaluate_retrieval_quality(
                query_results,
                test_case['expected_topics'],
                test_case['expected_source']
            )
            
            results.append({
                'question': test_case['question'],
                'category': test_case['category'],
                'metrics': metrics,
                'raw_results': query_results
            })
        
        # Cleanup
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        
        return {
            'config': config,
            'build_time': build_time,
            'results': results
        }
    
    def run_full_test(self) -> Dict:
        """Run tests on all configurations"""
        configs = [
            {"chunk_size": 500, "overlap": 100},   # 20% overlap
            {"chunk_size": 1000, "overlap": 200},  # 20% overlap  
            {"chunk_size": 1000, "overlap": 500},  # 50% overlap (current)
            {"chunk_size": 1500, "overlap": 300},  # 20% overlap
            {"chunk_size": 2000, "overlap": 400},  # 20% overlap
        ]
        
        all_results = {}
        
        for config in configs:
            config_name = f"chunk_{config['chunk_size']}_overlap_{config['overlap']}"
            all_results[config_name] = self.test_configuration(config)
        
        return all_results

def main():
    print("Starting RAG Performance Testing...")
    
    tester = RAGPerformanceTester()
    results = tester.run_full_test()
    
    # Generate summary report
    summary = format_results_summary(results)
    print("\n" + summary)
    
    # Save detailed results
    with open("rag_performance_results.txt", "w") as f:
        f.write(summary)
        f.write("\n\n=== Detailed Results ===\n\n")
        
        for config_name, config_results in results.items():
            f.write(f"\n{config_name}:\n")
            for result in config_results['results']:
                f.write(f"\nQuestion: {result['question']}\n")
                f.write(f"Category: {result['category']}\n")
                f.write(f"Metrics: {result['metrics']}\n")
                f.write("-" * 50 + "\n")
    
    print("\nDetailed results saved to 'rag_performance_results.txt'")

if __name__ == "__main__":
    main()