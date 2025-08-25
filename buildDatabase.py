from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import glob

data_path = "data/"

CHROMA_PATH = "chroma"

load_dotenv()  # This loads the .env file

def load_documents():
    # Load markdown files
    md_loader = DirectoryLoader(data_path, glob="*.md")
    md_docs = md_loader.load()
    
    # Load PDF files using PyPDFLoader and combine pages
    pdf_docs = []
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        # Combine all pages into one document to preserve context
        if pages:
            combined_text = "\n\n".join([page.page_content for page in pages])
            combined_doc = Document(
                page_content=combined_text, 
                metadata={"source": pdf_file}
            )
            pdf_docs.append(combined_doc)
    
    # Combine all documents
    all_docs = md_docs + pdf_docs
    print(f"Loaded {len(all_docs)} documents ({len(md_docs)} md, {len(pdf_files)} pdf)")
    return all_docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_database(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    print(f"saved {len(chunks)} chunks to db at {CHROMA_PATH}")

def main():
    docs = load_documents()
    chunks = split_text(docs)
    build_database(chunks)

if __name__ == "__main__":
    main()