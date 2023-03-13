import streamlit as st
from langchain import PromptTemplate
from langchain import OpenAI

template = """
    Below is an email that may be poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect

    Here are some example different tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you.
    - Informal: Went to Barcelona for the weekend. Lots to tell you.

    Here are some examples of words in different dialect:
    - American English: French fries, cotton candy, apartment, garbage, cookie
    - British English: chips, candyfloss, flag, rubbish, biscuit, green fingers

    Below is the email, tone and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}

    REWORDED EMAIL:
"""


prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template=template,
)

def Load_LLM():
    llm = OpenAI(temperature=0.5)
    return llm

llm = Load_LLM()

st.set_page_config(page_title="Globalize email", page_icon=":robot:")
st.header("Globalize text")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Some markdown with [Langchain](www.langchain.com) and [OpenAI](www.openai.com)")

# with col2:
#     st.image(image='TweetScreentshot.jpg', width=500, caption='https://twitter.com')

st.markdown("## Enter your email to convert")

def get_text():
    input_text = st.text_area(label="", placeholder="your email...", key="email_input")
    return input_text

col1, col2 = st.columns(2)

with col1:
    optional_tone=st.selectbox('Which tone would you like to have?',
                                ('Formal', 'Informal'))

with col2:
    option_dialect=st.selectbox('Which English dialect would you like?',
                                 ('Americal English', 'British English'))
    

email_input = get_text()
st.markdown("### Your converted email")

if (email_input):
    prompt_with_email=prompt.format(tone=optional_tone, dialect=option_dialect, email=email_input)
    formatted_email = llm(prompt_with_email)

    st.write(formatted_email)
