import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json

@st.cache_resource
def Load_LLM():
    llm = ChatOpenAI(temperature=0)
    return llm

verbose = True

system_template = """
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

"""

if (not verbose) :
    system_template += """
    Answer both intent and obfuscation in one word only, without any explanation.
    Return the answer as a valid json object having two parameters "intent" and "obfuscation".
    """


human_template="""
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
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    llm = Load_LLM()
    script_output_msg = llm(chat_prompt.format_prompt(ScriptType=optional_ScriptType, Script=script_input).to_messages())

    st.markdown("### Raw output")
    st.write(script_output_msg.content)

    if (not verbose) :
        response_json_ob = json.loads(script_output_msg.content)
        st.write("#### Classification: ", response_json_ob["intent"])
        st.write("#### Obfuscated: ", response_json_ob["obfuscation"])

