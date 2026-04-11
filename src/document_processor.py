import pdfplumber
import docx
import os

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_pdf(self):
        if not self.file_path.endswith('.pdf'):
            raise ValueError('File is not a PDF')
        with pdfplumber.open(self.file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += (page.extract_text() or '') + '\n'
        return text

    def process_docx(self):
        if not self.file_path.endswith('.docx'):
            raise ValueError('File is not a DOCX')
        doc = docx.Document(self.file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += (paragraph.text or '') + '\n'
        return text

    def process_txt(self):
        if not self.file_path.endswith('.txt'):
            raise ValueError('File is not a TXT')
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

# Usage example:
# processor = DocumentProcessor('path/to/your/file.pdf')
# print(processor.process_pdf())