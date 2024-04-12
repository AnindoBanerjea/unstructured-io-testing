import streamlit as st

from utils import show_navigation
show_navigation()

st.write("# Welcome to Streamlit! 👋")

st.markdown(
  """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
 """
)
