import PyPDF2
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile


def read_file(uploaded_file: UploadedFile) -> str:
    """
    Reads the content of the uploaded file based on its file extension.

    Args:
        uploaded_file (UploadedFile): The file to be read.

    Returns:
        str: The content of the file as a string.
    """
    file_extension = get_file_extension(uploaded_file.name)
    if file_extension == "md":
        return read_markdown(uploaded_file)
    elif file_extension == "txt":
        return read_text(uploaded_file)
    elif file_extension == "pdf":
        return read_pdf(uploaded_file)
    elif file_extension == "csv":
        return read_csv(uploaded_file)
    elif file_extension == "arxiv":
        return read_arxiv(uploaded_file)
    else:
        return "Unsupported file type."


def read_text(uploaded_file: UploadedFile) -> str:
    """
    Read the contents of an uploaded file and return it as a string.

    Args:
        uploaded_file (UploadedFile): The uploaded file object.

    Returns:
        str: The contents of the uploaded file as a string.
    """
    return uploaded_file.getvalue().decode("utf-8")


def read_markdown(uploaded_file: UploadedFile) -> str:
    """
    Read the contents of a markdown file and return it as a string.

    Args:
        uploaded_file (UploadedFile): The uploaded markdown file.

    Returns:
        str: The contents of the markdown file as a string.
    """
    return read_text(uploaded_file)


def read_pdf(uploaded_file: UploadedFile) -> str:
    """
    Read the text content of a PDF file.

    Args:
        uploaded_file (UploadedFile): The PDF file to be read.

    Returns:
        str: The extracted text content of the PDF file.
    """
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() + "\n"
    return text


def create_sentences(df: pd.DataFrame) -> str:
    """
    This function creates readable sentences from a list of dataframes.

    Args:
        df (pd.DataFrame): The dataframe to be converted to sentences.

    Returns:
        str: The sentences created from the dataframe.
    """

    # Get the header of the dataframe as a list
    header = df.columns.tolist()

    # Initialize an empty string
    sentences = ""

    # Loop accross the rows of the dataframe
    for _, row in df.iterrows():
        # Convert the current row to a list
        row = row.tolist()

        for j, (header_element, row_element) in enumerate(zip(header, row)):
            if row_element == "NaN":
                row_element = "None"

            # Add the element to the sentence
            element = f"{header_element}: {row_element}, "
            sentences += element + "\n"

            # Remove the comma and add a new line if it's the last element
            if j == len(row) - 1:
                sentences = sentences[:-2] + ".\n"

    return sentences


def read_csv(uploaded_file: UploadedFile) -> str:
    """
    Reads a CSV file and returns its content as a string.

    Parameters:
        uploaded_file (UploadedFile): The uploaded CSV file.

    Returns:
        str: The content of the CSV file as a string.
    """
    df = pd.read_csv(uploaded_file)
    return create_sentences(df)


def read_arxiv(uploaded_file: UploadedFile) -> str:
    """
    Reads an ArXiv file.

    Args:
        uploaded_file (UploadedFile): The ArXiv file to be read.

    Returns:
        str: The content of the ArXiv file.
    """
    return read_pdf(uploaded_file)


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a given filename.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The file extension.
    """
    return filename.split(".")[-1]
