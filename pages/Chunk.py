import streamlit as st
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from openai import OpenAI
import json

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV=st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']

from utils import show_navigation
show_navigation()

client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"]) 

def embed(sections,filename):
    index = pc.Index(PINECONE_INDEX_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    idx = 0
    file_location = 0
    for text in sections:
        docs=text_splitter.create_documents([text])
        for d in docs:
            hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
            l = len(d.page_content)
            embedding=client.embeddings.create(model="text-embedding-ada-002", input=d.page_content).data[0].embedding
            metadata={"hash":hash,"section_text":text,"text":d.page_content,"index":idx,"location":file_location,"size":l,"model":"text-embedding-ada-003","docname":filename}
            file_location += l
            idx += 1
            index.upsert([(hash,embedding,metadata)])
    return


#
# Main
#

st.markdown("# Chunk processed Response file")

with open("RFP_sections.json", "r") as f:
    sections = json.load(f)
    embed(sections,"RFP_sections.json")
    st.write("Finished.")