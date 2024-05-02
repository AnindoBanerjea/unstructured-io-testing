import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from utils import show_navigation
show_navigation()

def get_model():
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets['OPENAI_API_KEY'])
    return model

def process_query(table_data):
    model=get_model()
    template = """Answer the question based only on the following context:
            {context}

            Question: Give the verbatim and full list of criteria that must be answered or requirements that must be met. Do not repeat the question in your answer.
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain =  prompt | model | output_parser
    resp=chain.invoke({"context": table_data})
    return resp

#
# Main
#

st.markdown("# Extract requirements from saved processed RFP file")

with open("RFP_requirements.txt", "r") as f:
    file_contents = f.read()
    final_resp=process_query(file_contents)
    st.write(final_resp)