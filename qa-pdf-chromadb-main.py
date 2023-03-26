import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ChatVectorDBChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from langchain.document_loaders import PagedPDFSplitter

@st.cache_resource
def create_qa():
    data_path = "/Users/Japneet/Documents/datasets/pdfs/"
    collection_name="pdf_embeddings"
    persist_directory="/Users/Japneet/Documents/datasets/pdfs/chromadb"

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(collection_name, embeddings, persist_directory=persist_directory)
#    chain = VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)
    llm = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')
    chain = ChatVectorDBChain.from_llm(llm, chain_type="stuff", vectorstore=vectorstore,
                                       top_k_docs_for_context=5, return_source_documents=True)
    return vectorstore, chain



st.set_page_config(page_title="Q&A on PDF", page_icon=":robot:")
st.header("Ask questions from PDF")

def get_text():
    input_text = st.text_area(label="input", placeholder="your query...", key="query_input")
    return input_text    

# def set_text(contents):
#     st.text_area(label="output", placeholder=contents, key="query_output")
    

vectorstore, qa = create_qa()
query_input = get_text()

st.markdown("### Your query output")

if (query_input):
    #query_output = qa.run(query_input)
    # docs = docsearch.similarity_search(query=query_input, include_metadata=True)
    # query_output = qa.run(input_documents=docs, question=query_input)
    #st.write(query_output.extra_info['sql_query'])
    #st.write(query_output["output"])
    # set_text(query_output)
    query_output=qa({"question": query_input, "chat_history": []})
    st.write(query_output)



