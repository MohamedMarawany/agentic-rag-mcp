# Agentic RAG MCP

A modular Retrieval-Augmented Generation (RAG) system with support for enhanced chunking, local and cloud LLMs, database integration, and multi-language QA (English/Arabic).

## Features
- Enhanced document chunking (Docling or custom)
- PDF ingestion and QA
- Local generative LLM support (Ollama: Mistral, Llama2, etc.)
- OpenAI GPT fallback (if API key provided)
- Local extractive QA (Transformers, English & Arabic)
- GraphDB, SQL, and NoSQL integration (placeholders)
- Modular, extensible codebase

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
   - Update the filename in `main.py` if needed.

4. **Run the app:**
   ```sh
   python main.py
   ```
   - Enter your question when prompted.
   - The system will use OpenAI, Ollama, or Transformers (in that order) and print which model answered.

## Configuration
- Edit `config.yaml` for DB and chunking settings.
- `.gitignore` is set up to exclude data, models, and secrets.

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies

## Notes
- For best local LLM results, use Ollama with a supported model (Mistral, Llama2, etc.).
- For Arabic QA, the system auto-selects an Arabic model if the question is in Arabic.
- GraphDB, SQL, and NoSQL modules are included as stubs for future expansion.

## License
MIT
