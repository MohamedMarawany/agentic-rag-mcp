# PDF Loader utility
import PyPDF2

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_text(self):
        text = ""
        with open(self.file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
