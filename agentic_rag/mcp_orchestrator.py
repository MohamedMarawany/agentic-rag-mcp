
import logging
from typing import Any, Callable, Optional, Dict
from agentic_rag.chunking import Chunker
from agentic_rag.retriever import retrieve_top_chunks
from agentic_rag.validation import AnswerValidator

logger = logging.getLogger(__name__)

class MCPOrchestrator:
    """
    Orchestrates the RAG pipeline using MCP: chunking, retrieval, LLM, and validation.
    """
    def __init__(self, chunker: Optional[Chunker] = None, validator: Optional[AnswerValidator] = None, llm_func: Optional[Callable[[str, str], Any]] = None):
        """
        Initialize the MCPOrchestrator.
        Args:
            chunker (Optional[Chunker]): Chunker instance.
            validator (Optional[AnswerValidator]): Validator instance.
            llm_func (Optional[Callable]): LLM function (context, question) -> (answer, model_name).
        """
        self.chunker = chunker or Chunker()
        self.validator = validator or AnswerValidator()
        self.llm_func = llm_func

    def run(self, document_text: str, question: str, top_n: int = 2) -> Dict[str, Any]:
        """
        Run the MCP pipeline: chunking, retrieval, LLM, and validation.
        Args:
            document_text (str): The input document text.
            question (str): The user's question.
            top_n (int): Number of top chunks to retrieve.
        Returns:
            Dict[str, Any]: Pipeline results including question, context, answer, model, and validation status.
        """
        try:
            # Step 1: Chunking
            chunks = self.chunker.chunk(document_text)
            # Step 2: Retrieval
            top_chunks = retrieve_top_chunks(chunks, question, top_n=top_n)
            context = ' '.join(top_chunks) if top_chunks else (chunks[0] if chunks else '')
            # Step 3: LLM answer
            if self.llm_func:
                answer, model_name = self.llm_func(context, question)
            else:
                answer, model_name = ("No LLM function provided", "None")
            # Step 4: Validation
            valid = self.validator.validate(answer)
            logger.info(f"MCPOrchestrator run complete. Model: {model_name}, Valid: {valid}")
            return {
                'question': question,
                'context': context,
                'answer': answer,
                'model': model_name,
                'valid': valid
            }
        except Exception as e:
            logger.error(f"Error in MCPOrchestrator run: {e}")
            return {
                'question': question,
                'context': '',
                'answer': '',
                'model': None,
                'valid': False,
                'error': str(e)
            }
