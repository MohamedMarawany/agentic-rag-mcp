from pydantic import validator
import yaml
from agentic_rag.chunking import Chunker
from agentic_rag.graphdb import GraphDB
from agentic_rag.database import SQLDatabase, NoSQLDatabase
from agentic_rag.ai_agent import WebSearchAgent
from agentic_rag.mcp_orchestrator import MCPOrchestrator
from agentic_rag.validation import AnswerValidator
from agentic_rag.arabic_benchmark import ArabicRAGBenchmark
from agentic_rag.pdf_loader import PDFLoader
from agentic_rag.retriever import retrieve_relevant_chunk
import openai
import os
import numpy as np
from typing import List

# NEW: HuggingFace Transformers for local QA
from transformers import pipeline
from langdetect import detect

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Load config
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ask_with_llm(chunk, question, openai_api_key):
    try:
        openai.api_key = openai_api_key
        prompt = f"Answer the following question based on the provided text.\n\nText:\n{chunk}\n\nQuestion: {question}\nAnswer:"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=256,
            temperature=0.2
        )
        return response.choices[0].text.strip(), "OpenAI"
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        if OLLAMA_AVAILABLE:
            return ask_with_ollama(chunk, question, model="mistral")
        else:
            return ask_with_transformers(chunk, question)

# NEW: Local Transformers QA with language detection
_transformers_qa_en = None
_transformers_qa_ar = None
def ask_with_transformers(chunk, question):
    global _transformers_qa_en, _transformers_qa_ar
    try:
        lang = detect(question)
    except Exception:
        lang = 'en'
    if lang == 'ar':
        if _transformers_qa_ar is None:
            _transformers_qa_ar = pipeline("question-answering", model="asafaya/bert-base-arabic-qa")
        qa = _transformers_qa_ar
    else:
        if _transformers_qa_en is None:
            _transformers_qa_en = pipeline("question-answering", model="deepset/roberta-base-squad2")
        qa = _transformers_qa_en
    result = qa(question=question, context=chunk)
    return result['answer'], "Transformers QA"

def ask_with_ollama(context, question, model="mistral"):
    if not OLLAMA_AVAILABLE:
        print("Ollama Python package not installed. Run 'pip install ollama'.")
        return None, "Ollama (not available)"
    prompt = f"Answer the following question based on the provided text.\n\nText:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.generate(model=model, prompt=prompt)
    return response['response'].strip(), "Ollama"

def split_into_chunks(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def retrieve_top_chunks(chunks: List[str], question: str, top_n: int = 2) -> List[str]:
    question_keywords = set(question.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_keywords & chunk_words)
        scores.append(score)
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [chunks[i] for i in top_indices if scores[i] > 0]

def main():
    config = load_config()
    # Chunking
    chunker = Chunker(method=config['chunking']['method'])
    # GraphDB
    graphdb = GraphDB(config['graphdb']['uri'], config['graphdb']['user'], config['graphdb']['password'])
    # SQL DB
    sqldb = SQLDatabase(config['sqldb']['uri'])
    # NoSQL DB
    nosqldb = NoSQLDatabase(config['nosqldb']['uri'])
    # Web Agent
    agent = WebSearchAgent() if config['web_agent']['enabled'] else None
    # Validator (fix: define before use)
    validator = AnswerValidator()
    # MCP Orchestrator with LLM selection logic
    def llm_func(context, question):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            return ask_with_llm(context, question, openai_api_key)
        elif OLLAMA_AVAILABLE:
            return ask_with_ollama(context, question, model="mistral")
        else:
            return ask_with_transformers(context, question)

    orchestrator = MCPOrchestrator(chunker=chunker, validator=validator, llm_func=llm_func)

    # PDF loading demo
    pdf_path = r'files\questions_english.pdf'  # Change to your PDF file name
    pdf_loader = PDFLoader(pdf_path)
    pdf_text = pdf_loader.load_text()
    question = input("Enter your question about the PDF: ")
    print(f"Q: {question}")
    result = orchestrator.run(pdf_text, question, top_n=2)
    print(f"[MCP] Model: {result['model']}")
    # print(f"[MCP] Context: {result['context'][:300]}...")
    print(f"[MCP] Answer: {result['answer']}")
    print(f"[MCP] Valid: {result['valid']}")

    # Example usage
    # doc = "Your document text here."
    # chunks = chunker.chunk(doc)
    # print(f"Chunks: {chunks}")
    # Add a node to GraphDB as a demo
    # graphdb.add_node('Document', {'text': doc})
    # Web search demo
    # if agent:
    #     print(agent.search('What is RAG?'))
    # # Validate answer
    # print(validator.validate('Sample answer'))
    # # Benchmark demo
    # print(benchmark.evaluate('Sample output'))
    graphdb.close()

if __name__ == "__main__":
    main()
