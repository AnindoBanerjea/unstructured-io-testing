from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV=st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']
client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

from utils import show_navigation
show_navigation()

def augmented_content(inp):
    # Create the embedding using OpenAI keys
    # Do similarity search using Pinecone
    # Return the top 1 results
    embedding=client.embeddings.create(model="text-embedding-ada-002", input=inp).data[0].embedding
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    results=index.query(vector=embedding,top_k=1,namespace="",include_metadata=True)
    rr=[ r['metadata']['section_text'] for r in results['matches']]
    return rr

company = "eCivis"

SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a business development expert responding to a government RFP. Utilize prior RFP examples wherever possible."
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Input a requirement."):
    retreived_content = augmented_content(prompt)
    prompt_guidance=f"""
Here is a previous RFP Response example for additional context:
{retreived_content}
Generate a thoughtful and detailed response for the following requirement based on the previous RFP
Response example and comparable in length. For the company name, use {company}. Do not invent company 
names. Do not format the response as a letter. Do not provide a preamble like 'Here is your response.'
"""
    st.session_state.messages.append({"role": "system", "content": prompt_guidance})
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):        
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages], 
            stream=True)
        response = st.write_stream(stream)

    with st.sidebar.expander("Retreival context provided to GPT-3"):
        st.write(f"{prompt_guidance}")
    st.session_state.messages.append({"role": "assistant", "content": response})