import streamlit as st

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import pinecone
import os

@st.cache_resource
def create_qa():
    data_path = "/Users/Japneet/Documents/datasets/pdfs/"

    loader = UnstructuredPDFLoader(data_path + "CBSE-Class-9-NCERT-Book-Science-MATTER-IN-OUR-SURROUNDINGS-chapter-1.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    pinecone.init(
        api_key='cefe6650-84ee-400a-a5bf-fb7c937254d2',
        environment='us-east-1-aws'
    )
    index_name='test1'
    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
#    chain = VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    return docsearch, chain    



st.set_page_config(page_title="Q&A on Logs", page_icon=":robot:")
st.header("Ask questions from logs")

def get_text():
    input_text = st.text_area(label="", placeholder="your query...", key="query_input")
    return input_text    

docsearch, qa = create_qa()
query_input = get_text()

st.markdown("### Your query output")

if (query_input):
    #query_output = qa.run(query_input)
    docs = docsearch.similarity_search(query_input, include_metadata=True)
    query_output = qa.run(input_documents=docs, question=query_input)
    #st.write(query_output.extra_info['sql_query'])
    #st.write(query_output["output"])
    st.write(query_output)

