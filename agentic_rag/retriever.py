# Simple keyword-based retriever for chunks
from typing import List

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
