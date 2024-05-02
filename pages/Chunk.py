import streamlit as st
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from openai import OpenAI

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV=st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']

from utils import show_navigation
show_navigation()

client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"]) 

def embed(text,filename):
    index = pc.Index(PINECONE_INDEX_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    docs=text_splitter.create_documents([text])
    file_location = 0
    for idx,d in enumerate(docs):
        hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        l = len(d.page_content)
        embedding=client.embeddings.create(model="text-embedding-ada-002", input=d.page_content).data[0].embedding
        metadata={"hash":hash,"text":d.page_content,"index":idx,"location":file_location,"size":l,"model":"text-embedding-ada-003","docname":filename}
        file_location += l
        index.upsert([(hash,embedding,metadata)])
    return


#
# Main
#

st.markdown("# Chunk processed Response file")

with open("RFP_response.txt", "r") as f:
    file_contents = f.read()
    embed(file_contents,"RFP_response.txt")
    st.write("Finished.")