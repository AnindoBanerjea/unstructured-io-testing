import time

import numpy as np

from openai import OpenAI
import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec, PodSpec

from utils import show_navigation
show_navigation()

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV=st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']

client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def augmented_content(inp):
    # Create the embedding using OpenAI keys
    # Do similarity search using Pinecone
    # Return the top 5 results
    embedding=client.embeddings.create(model="text-embedding-ada-002", input=inp).data[0].embedding
    pc = Pinecone(api_key=PINECONE_API_KEY)
    #pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pc.Index(PINECONE_INDEX_NAME)
    results=index.query(vector=embedding,top_k=3,namespace="",include_metadata=True)
    #print(f"Results: {results}")
    with st.sidebar.expander("RAG Augmentation"):
        st.write(f"Results: {results}")
    rr=[ r['metadata']['text'] for r in results['matches']]
    #print(f"RR: {rr}")
    #st.write(f"RR: {rr}")
    return rr


SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a business development expert responding to a government RFP. Utilize prior RFP examples whereever possible."
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    retreived_content = augmented_content(prompt)
    #print(f"Retreived content: {retreived_content}")
    prompt_guidance=f"""
Please guide the user with the following information:
{retreived_content}
The user's question was: {prompt}
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        messageList=[{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages]
        messageList.append({"role": "user", "content": prompt_guidance})
        
        for response in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messageList, stream=True):
            delta_response=response.choices[0].delta
            print(f"Delta response: {delta_response}")
            if delta_response.content:
                full_response += delta_response.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    with st.sidebar.expander("Retreival context provided to GPT-3"):
        st.write(f"{retreived_content}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})