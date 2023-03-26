import openai
import chromadb
import langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from langchain.document_loaders import PagedPDFSplitter

def ingest():
    data_path = "/Users/Japneet/Documents/datasets/pdfs/"
    collection_name="pdf_embeddings"
    persist_directory="/Users/Japneet/Documents/datasets/pdfs/chromadb"

    # loader = UnstructuredPDFLoader(data_path + "CBSE-Class-9-NCERT-Book-Science-MATTER-IN-OUR-SURROUNDINGS-chapter-1.pdf")
    # data = loader.load()
    loader = PagedPDFSplitter(data_path + "CBSE-Class-9-NCERT-Book-Science-MATTER-IN-OUR-SURROUNDINGS-chapter-1.pdf")
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    docs_db = Chroma.from_documents(documents, embeddings, collection_name=collection_name, persist_directory=persist_directory)
    docs_db.persist()



ingest()