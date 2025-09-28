from agentic_rag.chunking import Chunker
from agentic_rag.retriever import retrieve_top_chunks
from agentic_rag.validation import AnswerValidator

# MCP-based orchestration (placeholder)
class MCPOrchestrator:
    def __init__(self, chunker=None, validator=None, llm_func=None):
        self.chunker = chunker or Chunker()
        self.validator = validator or AnswerValidator()
        self.llm_func = llm_func  # function(context, question) -> (answer, model_name)

    def run(self, document_text, question, top_n=2):
        # Step 1: Chunking
        chunks = self.chunker.chunk(document_text)
        # Step 2: Retrieval
        top_chunks = retrieve_top_chunks(chunks, question, top_n=top_n)
        context = ' '.join(top_chunks) if top_chunks else chunks[0]
        # Step 3: LLM answer
        if self.llm_func:
            answer, model_name = self.llm_func(context, question)
        else:
            answer, model_name = ("No LLM function provided", "None")
        # Step 4: Validation
        valid = self.validator.validate(answer)
        return {
            'question': question,
            'context': context,
            'answer': answer,
            'model': model_name,
            'valid': valid
        }
