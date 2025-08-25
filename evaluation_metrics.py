"""
Evaluation metrics for RAG system performance
"""
import re
from typing import List, Dict, Tuple
from langchain.schema import Document

def calculate_topic_overlap(retrieved_chunks: List[Document], expected_topics: List[str]) -> float:
    """
    Calculate how many expected topics appear in retrieved chunks
    Returns: ratio of expected topics found (0.0 to 1.0)
    """
    if not expected_topics:
        return 1.0  # If no topics expected, return perfect score
    
    # Combine all retrieved text
    combined_text = " ".join([doc.page_content.lower() for doc in retrieved_chunks])
    
    # Count how many expected topics are found
    found_topics = 0
    for topic in expected_topics:
        if topic.lower() in combined_text:
            found_topics += 1
    
    return found_topics / len(expected_topics)

def check_source_accuracy(retrieved_chunks: List[Document], expected_source: str) -> float:
    """
    Check if retrieved chunks come from the expected source document
    Returns: ratio of chunks from correct source (0.0 to 1.0)
    """
    if expected_source == "none":
        return 1.0  # For invalid questions, any source is acceptable
    
    if not retrieved_chunks:
        return 0.0
    
    correct_chunks = 0
    for doc in retrieved_chunks:
        source = doc.metadata.get('source', '')
        if expected_source in source:
            correct_chunks += 1
    
    return correct_chunks / len(retrieved_chunks)

def calculate_avg_relevance_score(results: List[Tuple[Document, float]]) -> float:
    """
    Calculate average relevance/similarity score from search results
    Returns: average score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    scores = [score for _, score in results]
    return sum(scores) / len(scores)

def evaluate_retrieval_quality(results: List[Tuple[Document, float]], expected_topics: List[str], expected_source: str) -> Dict[str, float]:
    """
    Comprehensive evaluation of retrieval results
    Returns: dictionary with all metrics
    """
    if not results:
        return {
            "topic_overlap": 0.0,
            "source_accuracy": 0.0, 
            "avg_relevance_score": 0.0,
            "num_chunks_retrieved": 0
        }
    
    chunks = [doc for doc, _ in results]
    
    return {
        "topic_overlap": calculate_topic_overlap(chunks, expected_topics),
        "source_accuracy": check_source_accuracy(chunks, expected_source),
        "avg_relevance_score": calculate_avg_relevance_score(results),
        "num_chunks_retrieved": len(chunks)
    }

def format_results_summary(all_results: Dict) -> str:
    """
    Format evaluation results into a readable summary
    """
    summary = "=== RAG Performance Test Results ===\n\n"
    
    for config_name, config_results in all_results.items():
        summary += f"Configuration: {config_name}\n"
        summary += f"{'='*50}\n"
        
        # Calculate averages across all questions
        metrics = ['topic_overlap', 'source_accuracy', 'avg_relevance_score']
        avg_metrics = {}
        
        for metric in metrics:
            scores = [result['metrics'][metric] for result in config_results['results']]
            avg_metrics[metric] = sum(scores) / len(scores) if scores else 0.0
        
        summary += f"Average Topic Overlap: {avg_metrics['topic_overlap']:.3f}\n"
        summary += f"Average Source Accuracy: {avg_metrics['source_accuracy']:.3f}\n" 
        summary += f"Average Relevance Score: {avg_metrics['avg_relevance_score']:.3f}\n"
        summary += f"Total Questions: {len(config_results['results'])}\n"
        summary += f"Build Time: {config_results.get('build_time', 'N/A')} seconds\n\n"
    
    return summary