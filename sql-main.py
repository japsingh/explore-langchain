import streamlit as st
from sqlalchemy import create_engine, MetaData
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

@st.cache_resource
def create_agent():
    engine = create_engine(
        'snowflake://{user}:@{account_identifier}/{database}/{schema_name}?authenticator={authenticator}&warehouse={warehouse}'.format(
            user='japneet.singh@lacework.net',
            account_identifier='LWDEV',
            database='japneet_test_db',
            schema_name='PUBLIC',
            authenticator='externalbrowser',
            warehouse='DEV_TEST',
        )
    )
    metadata_obj = MetaData(bind=engine)
    custom_table_info = {
        "process": """
                    "mid" : 'mid column provides machine id of the machine where the process is running. mid links process table with machine table.',
                    "start_time" : 'start_time column provides start time of the process',
                    "pid" : 'pid column provides process id of the process',
                    "pid_hash" : 'pid_hash column provides unique hash which uniquely identifies the process',
                    "ppid" : 'ppid column represents process id of the parent process of the process',
                    "ppid_hash" : 'ppid_hash column provides unique hash which uniquely identifies the parent process of the process',
                    "username" : 'username column provides info on the user context under which the process is running'
                    "EXE_PATH" : 'EXE_PATH column contains full file path of the process, including process name',
                    "cmdline_hash" : 'cmdline_hash column is a foreign key which links process table with cmdline table. This column helps find command line of a process.',
                    
                    """
    }
    db = SQLDatabase(engine, custom_table_info=custom_table_info)
    toolkit = SQLDatabaseToolkit(db=db)

    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True,
        return_intermediate_steps=True
    )
    return agent_executor



st.set_page_config(page_title="Execute Database queries in natural language", page_icon=":robot:")
st.header("Threat hunting with natural language")

agent_executor = create_agent()

st.markdown("## Enter your query") 

def get_text():
    input_text = st.text_area(label="", placeholder="your query...", key="query_input")
    return input_text    

query_input = get_text()
st.markdown("### Your query output")

intermediate_steps = ""

if (query_input):
    query_output = agent_executor({query_input})
    #st.write(query_output.extra_info['sql_query'])
    st.write(query_output["output"])
    intermediate_steps = query_output["intermediate_steps"]

st.markdown("### Generated SQL")
if (intermediate_steps):
    final_sql = intermediate_steps[-1][0][1]
    st.write(final_sql)