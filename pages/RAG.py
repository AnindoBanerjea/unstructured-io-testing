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
    #print(f"Results: {results}")
    #with st.sidebar.expander("RAG Augmentation"):
    #    st.write(f"Results: {results}")
    rr=[ r['metadata']['text'] for r in results['matches']]
    result_index = 0
    for r in results['matches']:
        idx = r['metadata']['index'] 
        nextresult = index.query(vector=embedding,filter={"index": idx+1},top_k=1,namespace="",include_metadata=True)
        #print(f"Index+1: {nextresult}")
        rr[result_index] += nextresult['matches'][0]['metadata']['text']
        nextresult = index.query(vector=embedding,filter={"index": idx+2},top_k=1,namespace="",include_metadata=True)
        #print(f"Index+2: {nextresult}")
        rr[result_index] += nextresult['matches'][0]['metadata']['text']
        result_index += 1

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

if prompt := st.chat_input("Input a requirement to generate a response to?"):
    retreived_content = augmented_content(prompt)
    #print(f"Retreived content: {retreived_content}")
    prompt_guidance=f"""
Here is previous RFP Response example:
{retreived_content}
Do not format the response as a letter. This response will go into a list of many responses. Do not generate any header or footer. Generate a response for the following requirement: {prompt}
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
        st.write(f"{prompt_guidance}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})