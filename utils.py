# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap

import streamlit as st

def show_navigation():
    with st.container(border=True):
        col1,col2,col3,col4,col5,col6=st.columns(6)
        col1.page_link("Hello.py", label="Home", icon="🏠")
        col2.page_link("pages/Upload_RFP.py", label="RFP", icon="1️⃣")
        col3.page_link("pages/Requirements.py", label="Reqs", icon="2️⃣")
        col4.page_link("pages/Upload_Response.py", label="Response", icon="3️⃣")
        col5.page_link("pages/Chunk.py", label="Chunk", icon="4️⃣")
        col6.page_link("pages/RAG.py", label="RAG", icon="5️⃣")
        #cols=st.columns(len(navList)
        # col3.page_link("pages/1_chat_with_AI.py", label="Chat", icon="2️⃣", disabled=True)
