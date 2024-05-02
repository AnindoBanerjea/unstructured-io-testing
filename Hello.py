import streamlit as st

from utils import show_navigation
show_navigation()

st.write("# Welcome to RFP Response Assistant!")

st.markdown(
  """
        A quick demo to show RFP parsing, requirements extraction, and response generation based on uploaded previous responses.
 """
)
