
# PDF Loader utility
import logging
from typing import Optional
import PyPDF2

logger = logging.getLogger(__name__)

class PDFLoader:
    """
    Utility class for loading and extracting text from PDF files.
    """
    def __init__(self, file_path: str):
        """
        Initialize the PDFLoader.
        Args:
            file_path (str): Path to the PDF file.
        """
        self.file_path = file_path

    def load_text(self) -> Optional[str]:
        """
        Extract text from the PDF file.
        Returns:
            Optional[str]: The extracted text, or None if extraction fails.
        """
        text = ""
        try:
            with open(self.file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            logger.info(f"Successfully loaded text from {self.file_path}")
            return text
        except Exception as e:
            logger.error(f"Failed to load PDF {self.file_path}: {e}")
            return None
