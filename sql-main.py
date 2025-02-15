import streamlit as st
from sqlalchemy import create_engine, MetaData
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
import pandas as pd

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
                    "mid" : 'mid column provides machine id of the machine where the process is running. mid links process table with machine table. mid is useful for joins with other tables.',
                    "start_time" : 'start_time column provides start time of the process',
                    "pid" : 'pid column provides process id of the process, this column doesn't identify process uniquely as there can be duplicate pid for multiple processes',
                    "pid_hash" : 'pid_hash column provides unique hash which uniquely identifies the process',
                    "ppid" : 'ppid column represents process id of the parent process of the process',
                    "ppid_hash" : 'ppid_hash column provides unique hash which uniquely identifies the parent process of the process',
                    "username" : 'username column provides info on the user context under which the process is running'
                    "EXE_PATH" : 'EXE_PATH column contains full file path of the process, including process name. Anytime "processes" or "process" is mentioned in the query, it refers to exe_path',
                    "cmdline_hash" : 'cmdline_hash column is a foreign key which links process table with cmdline table. This column helps find command line of a process.',
                    
                    """,
        "machine": """
                    "mid": 'mid column provides machine id of the machine where the process is running. mid links machine table with other tables.',
                    "hostname": 'hostname column provides the machine name / host name of the machine",
                    "os": 'os column provides Operating system installed on the machine, it can be one of 'Linux' or 'Windows'
                   """
    }
    db = SQLDatabase(engine, custom_table_info=custom_table_info)
    toolkit = SQLDatabaseToolkit(db=db)

    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        toolkit=toolkit,
        verbose=True,
        return_intermediate_steps=True,
        top_k=1024
    )
    return agent_executor, db



st.set_page_config(page_title="Execute Database queries in natural language", page_icon=":robot:")
st.header("Visibility and Insights using natural language")

agent_executor, db = create_agent()

st.markdown("## Enter your query") 

def get_text():
    input_text = st.text_area(label="", placeholder="your query...", key="query_input")
    return input_text

query_input = get_text()
st.markdown("### Your query output")

intermediate_steps = ""

if (query_input):
    try:
        query_output = agent_executor({query_input})
        #st.write(query_output.extra_info['sql_query'])
        st.write(query_output["output"])
        intermediate_steps = query_output["intermediate_steps"]
    except:
        st.write("Can't get answer to the query")

if (intermediate_steps):
    st.markdown("### Generated SQL")
    try:
        final_sql = intermediate_steps[-1][0][1]
        st.write(final_sql)
    except:
        st.write("Can't obtain SQL query")

        # st.write(intermediate_steps)
        
    try:
        with db._engine.begin() as connection:
            st.markdown("### Output in Tabular form")
            if db._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {db._schema}")
            cursor = connection.execute(final_sql)
            if (len(cursor.keys()) != 0):
                # col_results = cursor.keys()
                # st.write(col_results)

                # list_results = cursor.fetchall()
                # st.write(list_results)

                df = pd.DataFrame(cursor.fetchall())
                adjustedCols = [key for key in cursor.keys() if key not in ['mid']]
                st.write(adjustedCols)
                df.columns = adjustedCols #[rec[0] for rec in cursor.description]
                st.table(df)
    except:
        st.write("Can't display output")
