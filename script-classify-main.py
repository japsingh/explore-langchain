import streamlit as st
from langchain import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import json

@st.cache_resource
def Load_LLM():
    llm = OpenAI(temperature=0)
    return llm


template = """
You are a Malware analyst.
Look at the below {ScriptType} script which is going to be executed on a system.
Your goal is to:
1. Deduce intent of the script
2. Deduce whether the script is Obfuscated or not

Different types of possible intents you can return:
1. 'malicious'
2. 'suspicious'
3. 'benign'
4. 'unknown'

Different types of possible obfuscation types you can return:
1. 'obfuscated'
2. 'not obfuscated'
3. 'unknown'

Please answer both intent and obfuscation in one word only.
Return the response as a json object having two parameters "intent" and "obfuscation".

{ScriptType} Script:
{Script}
"""

st.set_page_config(page_title="Script classification", page_icon=":robot:")
st.header("Classify scripts as malicious or benign")

st.markdown("## Enter the script") 

def get_text():
    input_text = st.text_area(label="", placeholder="script...", key="script_input")
    return input_text    

col1, col2 = st.columns(2)

with col1:
    optional_ScriptType=st.selectbox('What is the language of the input script?',
                                ('Powershell', 'Javascript', 'VBScript'))

script_input = get_text()
st.markdown("### Output")

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

if (script_input):
    prompt = PromptTemplate(
        input_variables=["ScriptType", "Script"],
        template=template,
    )

    llm = Load_LLM()
    prompt_with_script=prompt.format(ScriptType=optional_ScriptType, Script=script_input)
    script_output = llm(prompt_with_script)

    st.markdown("### Raw output")
    st.write(script_output)

    response_json_ob = json.loads(script_output)
    st.write("#### Classification: ", response_json_ob["intent"])
    st.write("#### Obfuscated: ", response_json_ob["obfuscation"])

