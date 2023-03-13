import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI

import magic
import os
import nltk
# from llama_index import GPTTreeIndex, SimpleDirectoryReader

#nltk.download('averaged_perceptron_tagger')

@st.cache_resource
def create_qa():
    data_path = "/Users/Japneet/Documents/datasets/logs/win-agent"

    with open(data_path + '/LWDataCollector_0.log') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])
    chain = VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)

    return chain



st.set_page_config(page_title="Q&A on Logs", page_icon=":robot:")
st.header("Ask questions from logs")

def get_text():
    input_text = st.text_area(label="", placeholder="your query...", key="query_input")
    return input_text    

qa = create_qa()
query_input = get_text()

st.markdown("### Your query output")

if (query_input):
    #query_output = qa.run(query_input)
    query_output = qa({"question": query_input}, return_only_outputs=True)
    #st.write(query_output.extra_info['sql_query'])
    #st.write(query_output["output"])
    st.write(query_output)

