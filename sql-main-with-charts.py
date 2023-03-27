# import os
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
import sys
import streamlit as st
import json
from sqlalchemy import create_engine, MetaData
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
import pandas as pd
from typing import List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.tools import BaseTool
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QueryCheckerTool,
    QuerySQLDataBaseTool,
    BaseSQLDatabaseTool,
)
from pydantic import BaseModel, Extra, Field, validator
from langchain.tools.base import BaseTool
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Execute Database queries in natural language", page_icon=":robot:")
st.header("Visibility and Insights using natural language")

class ChartCreatorTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for creating charts."""

    name = "create_pie_chart"
    description = """
    Input to this tool is a json object which contains the following json elements:
    N number of json elements signifying each sub value of a pie chart having the following json format:

    "name": Name of sub value,
    "value": sub value number which is lesser than total

    Output of this tool is a visual chart depicting the pie chart.
    """

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("ChartCreatorTool does not support async")

    def parse_json(self, json_s: str) -> str:
        return "success"

    # def parse_json(self, json_s: str) -> str:
    #     print("running create_pie_chart tool. json_str: " + json_s)
    #     # print("typeof jsont_s is " + type(json_s))
    #     try:
    #         pie_chart_json = json.loads(str(json_s))
    #         print("pie_chart_json is " + pie_chart_json)
    #     except json.JSONDecodeError as errexc2:
    #         error_str="Json decode error while parsing input json: "+json_s
    #         print(error_str)
    #         return error_str
    #     except TypeError as errexc1:
    #         error_str="Type error while parsing input json: "+json_s+", error: "
    #         print(error_str)
    #         tuples = sys.exc_info()
    #         print(tuples)
    #         return error_str
    #     except ValueError as errexc:
    #         error_str="Value error while parsing input json: "+json_s
    #         print(error_str)
    #         return error_str
    #     except Exception as exc:
    #         error_str="Exception while parsing input json: "+json_s
    #         print(error_str)
    #         return error_str
    #     except:
    #         error_str="Generic Exception while parsing input json: "+json_s
    #         print(error_str)
    #         return error_str
    #     return "success"

    def _run(self, json_s: str) -> str:
        """Parse the input json, create the pie chart and return 'success', or return an error message."""
        return self.parse_json(json_s)



class LWAssistantToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySQLDataBaseTool(db=self.db),
            InfoSQLDatabaseTool(db=self.db),
            ListSQLDatabaseTool(db=self.db),
            QueryCheckerTool(db=self.db),
            ChartCreatorTool(db=self.db)
        ]


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
                    """,
        "cmdline": """
                    "cmdline": 'cmdline column provides cmdline / commandline of a process. The cmdline includes process path, along with commandline parameters passed to that process
                    "cmdline_hash": 'cmdline_hash column is a foreign key which links process table with cmdline table. This column helps find command line of a process.'
                   """
    }
    db = SQLDatabase(engine, custom_table_info=custom_table_info)
    toolkit = LWAssistantToolkit(db=db)

    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
        toolkit=toolkit,
        verbose=True,
        return_intermediate_steps=True,
        top_k=1024
    )
    return agent_executor, db




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
    except Exception as e:
        st.write("Can't get answer to the query: "+str(e))

if (intermediate_steps):
    try:
        final_step = intermediate_steps[-1][0][0]
        print("final_step: "+final_step)
        if final_step == "query_sql_db":
            st.markdown("### Generated SQL")
            final_sql = intermediate_steps[-1][0][1]
            st.write(final_sql)
        elif final_step == "create_pie_chart":
            st.markdown("### Pie Chart")
            json_str = intermediate_steps[-1][0][1]
            json_str = json_str.replace("\'", "\"")
            print("json_str: "+json_str)
            pie_chart_json = json.loads(json_str)

            print(pie_chart_json)
            if pie_chart_json != None:
                print("Input json is ", pie_chart_json)
                print("typeof pie_chart_json ", type(pie_chart_json))
                labels=[]
                values=[]

                for name in pie_chart_json:
                    print(name)
                    if pie_chart_json[name] != 0:
                        labels.append(name + "(" + str(pie_chart_json[name]) + ")")
                        values.append(pie_chart_json[name])

                npvalues = np.array(values)
                fig1, ax1 = plt.subplots()
                ax1.pie(npvalues, labels=labels, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
    except Exception as e:
        st.write("Can't obtain SQL query: " + str(e))

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
