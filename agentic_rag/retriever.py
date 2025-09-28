# Simple keyword-based retriever for chunks
from typing import List
import numpy as np

def retrieve_relevant_chunk(chunks: List[str], question: str) -> str:
    question_keywords = set(question.lower().split())
    best_chunk = ''
    best_score = 0
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_keywords & chunk_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk if best_score > 0 else (chunks[0] if chunks else '')

def retrieve_top_chunks(chunks: List[str], question: str, top_n: int = 2) -> List[str]:
    question_keywords = set(question.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_keywords & chunk_words)
        scores.append(score)
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]
