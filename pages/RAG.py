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
    # Return the top 5 results
    embedding=client.embeddings.create(model="text-embedding-ada-002", input=inp).data[0].embedding
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    results=index.query(vector=embedding,top_k=1,namespace="",include_metadata=True)
    rr=[ r['metadata']['section_text'] for r in results['matches']]
    return rr


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

if prompt := st.chat_input("Input a requirement to generate a response to?"):
    retreived_content = augmented_content(prompt)
    prompt_guidance=f"""
Here is a previous RFP Response example:
{retreived_content}
Generate a thoughtful and detailed response for the following requirement based on the previous example and comparable in length. Ignore the requirement that is repeated above the response. Do not format the response as a letter: {prompt}
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
            if delta_response.content:
                full_response += delta_response.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    with st.sidebar.expander("Retreival context provided to GPT-3"):
        st.write(f"{prompt_guidance}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})