import streamlit as st

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from llama_index import GPTSimpleVectorIndex
from llama_index import LLMPredictor
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import pinecone
import os
from langchain.document_loaders import PagedPDFSplitter

# @st.cache_resource
def create_qa():
    data_path = "/Users/Japneet/Documents/datasets/pdfs/"
    shouldParse = False
    shouldSave = True
    saveFile = 'index.json'

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5))

    if not os.path.exists(saveFile):
        shouldParse = True

    if shouldParse:
        loader = PagedPDFSplitter(data_path + "CBSE-Class-9-NCERT-Book-Science-MATTER-IN-OUR-SURROUNDINGS-chapter-1.pdf")
        data = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(data)

        # This can be composable with other indexes in the future.
        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)
    else:
        index = GPTSimpleVectorIndex.load_from_disk(saveFile)

    if shouldSave and shouldParse:
        print("Saving index...")
        index.save_to_disk(saveFile)

    return index    



st.set_page_config(page_title="Q&A on PDF", page_icon=":robot:")
st.header("Ask questions from PDF")

def get_text():
    input_text = st.text_area(label="input", placeholder="your query...", key="query_input")
    return input_text    


docsearch = create_qa()
query_input = get_text()

st.markdown("### Your query output")

if (query_input):
    query_output = docsearch.query(query_input, response_mode='Compact')
    st.write(query_output)



