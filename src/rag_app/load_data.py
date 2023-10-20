from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def initialize_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size = chunk_size,
                        chunk_overlap  = chunk_overlap,
                        length_function = len,
                        is_separator_regex = False,
                    )
    return text_splitter

def load_split_html_file(html_file, text_splitter):
    loader = UnstructuredHTMLLoader(html_file)
    data = loader.load_and_split(text_splitter)
    return data

def load_split_pdf_file(pdf_file, text_splitter):
    loaded = PyPDFLoader(pdf_file)
    data = loaded.load_and_split(text_splitter)
    return data



