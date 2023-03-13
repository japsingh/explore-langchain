import streamlit as st

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain import VectorDBQA
# from langchain.llms.openai import OpenAI
# from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import magic
import os
import nltk
# from llama_index import GPTTreeIndex, SimpleDirectoryReader

#nltk.download('averaged_perceptron_tagger')

@st.cache_resource
def create_qa():
    # st.markdown("## Enter path where logs are stored") 
    # def get_folder():
    #     input_folder = st.text_input(label="", placeholder="logs folder...", key="input_folder")
    #     return input_folder    
    # data_path = get_folder()
    # if (not data_path):
    data_path = "/Users/Japneet/Documents/datasets/logs/win-agent"

    glob_pattern = "**/*.log"
    # loader = DirectoryLoader(data_path, glob=glob_pattern)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # docsearch = Chroma.from_documents(texts, embeddings)
    loader = TextLoader(data_path + '/LWDataCollector_0.log')
    qa = VectorstoreIndexCreator().from_loaders([loader])


    # qa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0), chain_type='stuff', vectorstore=docsearch)
    # documents = SimpleDirectoryReader(data_path).load_data()
    # index = GPTTreeIndex(documents)
    # index.save_to_disk('index.json')
    # # try loading
    # qa = GPTTreeIndex.load_from_disk('index.json')
    return qa



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
    query_output = qa.query(query_input)
    #st.write(query_output.extra_info['sql_query'])
    #st.write(query_output["output"])
    st.write(query_output)

