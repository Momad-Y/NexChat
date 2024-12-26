import PyPDF2
import pandas as pd

def read_pdf(uploaded_file):
    """Read text from a PDF file."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def read_csv(uploaded_file):
    """Read text from a CSV file."""
    df = pd.read_csv(uploaded_file)
    return df.to_string()

def read_arxiv(uploaded_file):
    """Read text from an arXiv paper (assuming it's a PDF)."""
    return read_pdf(uploaded_file)

def get_file_extension(filename):
    """Get the file extension of a given file."""
    return filename.split('.')[-1]