import streamlit as st
import sys
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
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Union

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""
    chatGptResponse: str = ""
    output_area: st.empty

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print("on_llm_start")


    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.chatGptResponse += token
        self.output_area.markdown(f'{self.chatGptResponse}')
        # print("received token: " + token)
        # st.write(token)
        # sys.stdout.write(token)
        # sys.stdout.flush()


    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print("on_llm_end")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        print("on_llm_error")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("on_chain_start")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("on_chain_end")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        print("on_chain_error")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        print("on_tool_start")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print("on_agent_action")
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        print("on_tool_end")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        print("on_tool_error")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        print("on_text")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        print("on_agent_finish")

#@st.cache_resource
def Load_LLM(_callback_manager):
    llm = ChatOpenAI(streaming=True,callback_manager=_callback_manager,temperature=0, verbose=True)
    return llm

system_template = """
You are a Malware analyst.
Look at the below {ScriptType} script which is going to be executed on a system.
Your goal is to:
1. Deduce intent of the script
2. Deduce whether the script is Obfuscated or not
3. If script is malicious or suspicious, determine what MITRE Attack technique is used in the script

Different types of possible intents you can return:
1. 'malicious'      (if the script contents give a strong hint that script can be used for malicious purposes)
2. 'suspicious'     (if the script contents give a somewhat weaker hint that script can be used for malicious purposes)
3. 'benign'         (if the script contents give no hint at all that script can be used for malicious purposes)
4. 'unknown'        (if looking at script contents it cannot be determined whether script is malicious or suspicous or benign)

Different types of possible obfuscation types you can return:
1. 'obfuscated'
2. 'not obfuscated'
3. 'unknown'

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
with col2:
    optional_Verbose=st.selectbox('Verbose output?',
                                ('False', 'True'))

script_input = get_text()
st.markdown("### Output")

verbose = False
if optional_Verbose == 'True':
    verbose = True

if (not verbose) :
    system_template += """
    Answer both intent and obfuscation in one word only, without any explanation.
    Answer MITRE Attack technique as a comma separate list of Technique IDs
    Return the answer as a valid json object having three parameters "intent", "obfuscation" and "mitre".
    """

output_box = st.empty()

streaming_callback_handler = StreamingCallbackHandler()
streaming_callback_handler.output_area = output_box
streaming_callback_manager = CallbackManager([streaming_callback_handler])

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

if (script_input):
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    llm = Load_LLM(streaming_callback_manager)
    script_output_msg = llm(chat_prompt.format_prompt(ScriptType=optional_ScriptType, Script=script_input).to_messages())

    # if verbose:
    #st.write(streaming_callback_handler.chatGptResponse)
    # else:
    #     st.write(script_output_msg.content)

    # if (not optional_Verbose) :
    #     response_json_ob = json.loads(script_output_msg.content)
    #     st.write("#### Classification: ", response_json_ob["intent"])
    #     st.write("#### Obfuscated: ", response_json_ob["obfuscation"])

