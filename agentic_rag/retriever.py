
# Simple keyword-based retriever for chunks
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

def retrieve_relevant_chunk(chunks: List[str], question: str) -> str:
    """
    Retrieve the most relevant chunk for a given question using keyword overlap.
    Args:
        chunks (List[str]): List of document chunks.
        question (str): The user's question.
    Returns:
        str: The most relevant chunk, or the first chunk if no match is found.
    """
    try:
        question_keywords = set(question.lower().split())
        best_chunk = ''
        best_score = 0
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_keywords & chunk_words)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        logger.info(f"Best chunk score: {best_score}")
        return best_chunk if best_score > 0 else (chunks[0] if chunks else '')
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunk: {e}")
        return chunks[0] if chunks else ''

def retrieve_top_chunks(chunks: List[str], question: str, top_n: int = 2) -> List[str]:
    """
    Retrieve the top-N relevant chunks for a given question using keyword overlap.
    Args:
        chunks (List[str]): List of document chunks.
        question (str): The user's question.
        top_n (int): Number of top chunks to return.
    Returns:
        List[str]: List of top-N relevant chunks.
    """
    try:
        question_keywords = set(question.lower().split())
        scores = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_keywords & chunk_words)
            scores.append(score)
        if not scores:
            return []
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_chunks = [chunks[i] for i in top_indices if scores[i] > 0]
        logger.info(f"Top chunk scores: {[scores[i] for i in top_indices]}")
        return top_chunks
    except Exception as e:
        logger.error(f"Error in retrieve_top_chunks: {e}")
        return chunks[:top_n] if chunks else []
