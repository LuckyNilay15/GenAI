from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)
st.header('Research Tool')

# user_input=st.text_input(
#     'Enter your prompt'
# )
paper_input=st.selectbox(
    'Select a topic',[
        'Food',
        'Sports',
        'Health',
        'Technology'
    ]
)
paper_style=st.selectbox(
    'Select a style',
    [
        'Formal',
        'Informal',
        'Technical',
        'Casual'
    ]
)
length_input=st.text_input(
    'Enter the length of the summary'
)
template=load_prompt('template.json')

if st.button('Summarize'):
    chain=template | model #if fstring use so can't chain since prompttemplate has tight coupling with langchain ecosystem.
    result=chain.invoke( {'paper_input':paper_input,
    'paper_style':paper_style,
    'length_input':length_input})

    # prompt=template.invoke({
    # 'paper_input':paper_input,
    # 'paper_style':paper_style,
    # 'length_input':length_input
    #  })
    # result=model.invoke(prompt)
    st.text(result.content)


    