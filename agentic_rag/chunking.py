import logging
from typing import List, Optional

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Docling import failed: {e}")
    DOCLING_AVAILABLE = False



class Chunker:
    """
    Chunker for splitting documents into semantically meaningful chunks.
    Supports Docling (if available) or robust fallback to NLTK-based chunking.
    Automatically falls back to NLTK if Docling produces only one or a very large chunk.
    """

    def __init__(self, method: str = "docling", chunk_size: int = 500) -> None:
        """
        Initialize the Chunker.
        Args:
            method (str): Chunking method ('docling' or 'naive').
            chunk_size (int): Target chunk size for fallback chunking.
        """
        self.method: str = method
        self.chunk_size: int = chunk_size
        self.logger = logging.getLogger(__name__)

        if self.method == "docling" and not DOCLING_AVAILABLE:
            self.logger.warning("Docling not installed. Falling back to naive splitting.")
            self.method = "naive"

        self.converter: Optional[DocumentConverter] = None
        if self.method == "docling":
            try:
                self.converter = DocumentConverter()
            except Exception as e:
                self.logger.error(f"Failed to initialize Docling converter: {e}")
                self.method = "naive"
                self.converter = None

    def chunk(self, document: str) -> List[str]:
        """
        Split the document into chunks using the selected method.
        Args:
            document (str): The input document as a string.
        Returns:
            List[str]: List of document chunks.
        """
        try:
            self.logger.info(f"Chunking document using method: {self.method}")

            # --- Docling chunking ---
            if self.method == "docling" and self.converter:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmp:
                    tmp.write(document)
                    tmp_path = tmp.name
                try:
                    result = self.converter.convert(tmp_path)
                    doc = result.document
                    chunks: List[str] = []
                    if hasattr(doc, 'sections'):
                        for section in doc.sections:
                            for para in getattr(section, 'paragraphs', []):
                                text = getattr(para, 'text', '').strip()
                                if text:
                                    chunks.append(text)
                    elif hasattr(doc, 'paragraphs'):
                        for para in doc.paragraphs:
                            text = getattr(para, 'text', '').strip()
                            if text:
                                chunks.append(text)
                    else:
                        chunks = [str(doc)]
                    self.logger.info(f"Docling produced {len(chunks)} chunks.")
                    # Hybrid: If only 1 chunk or chunk is very large, fallback to NLTK chunking
                    if len(chunks) > 1 and (len(chunks) == 1 and len(chunks[0]) <= 2 * self.chunk_size):
                        return chunks
                    else:
                        self.logger.warning("Docling produced only 1 chunk or a very large chunk. Falling back to NLTK chunking for better granularity.")
                finally:
                    os.unlink(tmp_path)

            # --- NLTK-based chunking ---
            try:
                import nltk
                nltk.data.find('tokenizers/punkt')
            except (ImportError, LookupError):
                try:
                    import nltk
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    self.logger.warning(f"NLTK not available or punkt download failed: {e}. Using legacy naive splitting.")
                    # Fallback: legacy naive splitting
                    chunks = [
                        document[i:i+self.chunk_size]
                        for i in range(0, len(document), self.chunk_size)
                    ]
                    self.logger.info(f"Naive splitting produced {len(chunks)} chunks.")
                    return chunks

            import nltk
            paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
            sentences: List[str] = []
            for para in paragraphs:
                sentences.extend(nltk.sent_tokenize(para))

            chunks: List[str] = []
            overlap = 2  # number of sentences to overlap
            i = 0
            while i < len(sentences):
                current_chunk = sentences[i]
                j = i + 1
                while j < len(sentences) and len(current_chunk) + len(sentences[j]) < self.chunk_size:
                    current_chunk += " " + sentences[j]
                    j += 1
                chunks.append(current_chunk.strip())
                i = j - overlap if (j - overlap) > i else j

            self.logger.info(f"NLTK-based chunking produced {len(chunks)} chunks.")
            return chunks

        except Exception as e:
            self.logger.error(f"Error during chunking: {e}")
            return []
