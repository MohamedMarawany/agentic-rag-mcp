# Agentic RAG MCP

A modular Retrieval-Augmented Generation (RAG) system with advanced chunking, multi-source LLMs, robust database integration, and multi-language QA (English/Arabic).

## Features
- **Hybrid document chunking:** Docling (Markdown-aware) with automatic fallback to NLTK for granular, RAG-friendly chunks.
- **PDF ingestion and QA:** Ask questions about any PDF, with transparent chunk preview.
- **Multi-source LLM support:** Local (Ollama: Mistral, Llama2, etc.), OpenAI GPT fallback, and Transformers (English & Arabic).
- **Web search fallback:** DuckDuckGo, Wikipedia, and Serper API for real-world answers when LLMs/PDFs lack information.
- **Database integration:** Stores Q&A in SQL (SQLite), NoSQL (MongoDB), and Neo4j GraphDB, with clear source tracking (e.g., "WebSearchAgent - Serper").
- **Node linking:** Q&A interactions are linked in Neo4j for session/sequence analysis.
- **Validation:** Rule-based and LLM-based answer validation, especially for time/date and banking questions.
- **Multi-language QA:** Auto-selects Arabic model for Arabic questions.
- **Extensible, production-ready codebase:** Modular design for easy feature addition.

## Quickstart

1. **Clone the repo and install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **(Optional) Install Ollama for local LLMs:**
   - Download from https://ollama.com/download and run a model:
     ```sh
     ollama run mistral
     ```

3. **Prepare your PDF:**
   - Place your PDF in the `files/` directory (e.g., `files/questions_english.pdf`).
   - Update the filename in `main.py` or via CLI if needed.

4. **Run the app:**
   ```sh
   python main.py
   ```
   - Enter your question when prompted.
   - The system will show chunk previews, answer source, and validation status.
   - All Q&A are saved to SQL, MongoDB, and Neo4j.

5. **Inspect your data:**
   - Use Neo4j Browser, SQLite tools, or MongoDB Compass to view Q&A records and relationships.

## Configuration
- Edit `config.yaml` for DB URIs, chunking method, and web agent settings.
- `.gitignore` excludes data files (e.g., `rag.db`), models, and secrets.

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies

## Notes
- For best local LLM results, use Ollama with a supported model (Mistral, Llama2, etc.).
- For Arabic QA, the system auto-selects an Arabic model if the question is in Arabic.
- All answer sources (LLM, WebSearchAgent) are tracked and stored.
- Neo4j nodes are linked for session/sequence analysis.
- If Docling chunking produces only one chunk, NLTK chunking is used automatically.

## Advanced Usage
- Run with CLI options for custom PDF, config, or to show all Q&A:
  ```sh
  python main.py --pdf files/questions_english.pdf --show-sql
  python main.py --show-nosql
  ```
- Export or compare Q&A records across databases for analytics.

## Troubleshooting
- If chunking fails, check Docling installation/version; fallback to NLTK is automatic.
- For OpenAI quota errors, use Ollama or Transformers.
- For missing records, check logs for SQL errors or use export scripts to compare databases.

---

For feature requests or help, open an issue or contact the maintainer.
