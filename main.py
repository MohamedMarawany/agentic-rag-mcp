
import argparse
import logging
import os
import yaml
from agentic_rag.chunking import Chunker
from agentic_rag.graphdb import GraphDB
from agentic_rag.database import SQLDatabase, NoSQLDatabase
from agentic_rag.ai_agent import WebSearchAgent
from agentic_rag.mcp_orchestrator import MCPOrchestrator
from agentic_rag.validation import AnswerValidator
from agentic_rag.arabic_benchmark import ArabicRAGBenchmark
from agentic_rag.pdf_loader import PDFLoader

def load_config(path: str = 'config.yaml') -> dict:
    """
    Load YAML configuration file.
    Args:
        path (str): Path to the config file.
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

def get_llm_func():
    """
    Returns a function that selects the best available LLM (OpenAI, Ollama, Transformers).
    """
    import openai
    try:
        import ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
    from transformers import pipeline
    from langdetect import detect
    _transformers_qa_en = None
    _transformers_qa_ar = None

    def ask_with_llm(chunk, question, openai_api_key):
        try:
            openai.api_key = openai_api_key
            # Improved prompt for time/date reasoning
            prompt = (
                "Answer the following question based on the provided text. "
                "If the question is about working hours, days, or times, reason step by step and be precise. "
                "\n\nText:\n{chunk}\n\nQuestion: {question}\nAnswer:"
            )
            prompt = prompt.format(chunk=chunk, question=question)
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=256,
                temperature=0.2
            )
            return response.choices[0].text.strip(), "OpenAI"
        except Exception as e:
            logging.warning(f"[OpenAI ERROR] {e}")
            if OLLAMA_AVAILABLE:
                return ask_with_ollama(chunk, question, model="mistral")
            else:
                return ask_with_transformers(chunk, question)

    def ask_with_transformers(chunk, question):
        nonlocal _transformers_qa_en, _transformers_qa_ar
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
            logging.warning("Ollama Python package not installed. Run 'pip install ollama'.")
            return None, "Ollama (not available)"
        import ollama
        prompt = f"Answer the following question based on the provided text.\n\nText:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = ollama.generate(model=model, prompt=prompt)
        return response['response'].strip(), "Ollama"

    def llm_func(context, question):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            return ask_with_llm(context, question, openai_api_key)
        elif OLLAMA_AVAILABLE:
            return ask_with_ollama(context, question, model="mistral")
        else:
            return ask_with_transformers(context, question)
    return llm_func

def main():
    """
    Main entry point for the Agentic RAG system. Handles CLI, PDF QA, and pipeline orchestration.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Agentic RAG System: PDF QA with LLM fallback and MCP orchestration.")
    parser.add_argument('--pdf', type=str, default=r'files\questions_english.pdf', help='Path to the PDF file to query.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--question', type=str, help='Question to ask about the PDF.')
    parser.add_argument('--show-sql', action='store_true', help='Display all stored Q&A from SQL database and exit.')
    parser.add_argument('--show-nosql', action='store_true', help='Display all stored Q&A from MongoDB and exit.')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        chunker = Chunker(method=config['chunking']['method'])
        graphdb = GraphDB(config['graphdb']['uri'], config['graphdb']['user'], config['graphdb']['password'])
        sqldb = SQLDatabase(config['sqldb']['uri'])
        nosqldb = NoSQLDatabase(config['nosqldb']['uri'])
        agent = WebSearchAgent() if config['web_agent']['enabled'] else None
        validator = AnswerValidator()
        llm_func = get_llm_func()
        orchestrator = MCPOrchestrator(chunker=chunker, validator=validator, llm_func=llm_func)

        # Show all Q&A from SQL or NoSQL and exit if requested
        if args.show_sql:
            # records = sqldb.get_all_qa()
            # print("\n--- All Q&A from SQL Database ---")
            # for rec in records:
            #     print(f"ID: {rec.get('id')} | Q: {rec.get('question')} | A: {rec.get('answer')} | Model: {rec.get('model')} | Valid: {rec.get('valid')} | Time: {rec.get('timestamp')} | File: {rec.get('source_file')}")
            return
        if args.show_nosql:
            # records = nosqldb.get_all_qa()
            # print("\n--- All Q&A from MongoDB ---")
            # for rec in records:
            #     print(f"Q: {rec.get('question')} | A: {rec.get('answer')} | Model: {rec.get('model')} | Valid: {rec.get('valid')} | Time: {rec.get('timestamp')} | File: {rec.get('source_file')}")
            return

        pdf_loader = PDFLoader(args.pdf)
        pdf_text = pdf_loader.load_text()
        if not pdf_text:
            logging.error(f"Failed to extract text from PDF: {args.pdf}")
            return
        
        # --- Show chunking step ---
        chunks = chunker.chunk(pdf_text)
        print("\n--- Chunking Preview ---")
        for i, ch in enumerate(chunks, 1):
            print(f"\n[Chunk {i}] {ch[:1000]}...")  # only show first 1000 chars
        print(f"\nTotal chunks created: {len(chunks)}")

        question = args.question or input("Enter your question about the PDF: ")
        logging.info(f"Question: {question}")

        result = orchestrator.run(pdf_text, question, top_n=2)
        # Fallback: If answer is not found in the PDF, use WebSearchAgent

        answer_text = result['answer'].strip()
        answer_lower = answer_text.lower()
        not_found_phrases = [
            'not found', 'does not contain', 'no information', 'no mention', 'cannot find',
            'no data', 'no answer', 'no relevant', 'no details', 'no record',
            'the text provided does not mention', 'no content', 'no result', 'irrelevant',
            'no such', 'no evidence', 'no reference', 'no context', 'no description',
            'no explanation', 'no such information', 'no such data', 'no such answer',
        ]
        generic_answers = [
            '', 'n/a', 'none', 'no', 'unknown', 'not applicable', 'not available', 'no response',
        ]
        use_web_search = False
        # Trigger fallback if answer is empty, generic, or contains any not-found phrase
        if answer_lower in generic_answers or not answer_text:
            use_web_search = True
        else:
            for phrase in not_found_phrases:
                if phrase in answer_lower:
                    use_web_search = True
                    break
        if use_web_search and agent is not None:
            print("[MCP] No relevant answer found in PDF or LLM. Using WebSearchAgent fallback...")
            logging.info("Answer not found in PDF or LLM. Using WebSearchAgent fallback.")
            web_answer, web_source = agent.search(question)
            result['answer'] = web_answer
            result['model'] = web_source
            print(f"[WebSearch] Answer: {web_answer}")
        print(f"[MCP] Model: {result['model']}")
        print(f"[MCP] Answer: {result['answer']}")
        print(f"[MCP] Valid: {result['valid']}")
        if 'error' in result:
            logging.error(f"Pipeline error: {result['error']}")

        # --- GraphDB Integration: Store Q&A interaction and link to previous ---
        try:
            from datetime import datetime
            node_properties = {
                'question': question,
                'answer': result.get('answer', ''),
                'context': result.get('context', ''),
                'model': result.get('model', ''),
                'valid': result.get('valid', False),
                'timestamp': datetime.utcnow().isoformat()
            }
            graphdb.add_node('QAInteraction', node_properties)

            # Link to previous QAInteraction node (by most recent timestamp)
            # (Assumes only one user/session is writing at a time)
            try:
                with graphdb.driver.session() as session:
                    prev = session.run(
                        """
                        MATCH (n:QAInteraction)
                        WHERE n.timestamp < $now
                        RETURN n.timestamp AS ts
                        ORDER BY n.timestamp DESC LIMIT 1
                        """,
                        now=node_properties['timestamp']
                    ).single()
                    if prev:
                        prev_ts = prev['ts']
                        graphdb.add_relationship(
                            from_label='QAInteraction', from_key='timestamp', from_value=prev_ts,
                            to_label='QAInteraction', to_key='timestamp', to_value=node_properties['timestamp'],
                            rel_type='NEXT'
                        )
            except Exception as e:
                logging.error(f"Failed to link QAInteraction nodes: {e}")
        except Exception as e:
            logging.error(f"Failed to store Q&A in GraphDB: {e}")

        # --- SQL/NoSQL Integration: Store Q&A interaction ---
        try:
            sqldb.save_qa(
                question=question,
                answer=result.get('answer', ''),
                model=result.get('model', ''),
                valid=result.get('valid', False),
                source_file=args.pdf
            )
        except Exception as e:
            logging.error(f"Failed to store Q&A in SQL DB: {e}")
        try:
            nosqldb.save_qa(
                question=question,
                answer=result.get('answer', ''),
                model=result.get('model', ''),
                valid=result.get('valid', False),
                source_file=args.pdf
            )
        except Exception as e:
            logging.error(f"Failed to store Q&A in MongoDB: {e}")

        graphdb.close()
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
