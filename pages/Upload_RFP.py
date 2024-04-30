import streamlit as st

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.staging.base import dict_to_elements




from utils import show_navigation
show_navigation()

def process_file(file_contents, file_name):
    s=UnstructuredClient(api_key_auth=st.secrets['UNSTRUCTURED_API_KEY'])

    files=shared.Files(
        content=file_contents,
        file_name=file_name,
    )

    req = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
    )

    try:
        resp = s.general.partition(req)
        elements = dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)

    tables = [el for el in elements if el.category == "Table"]
    final_text=""
    for el in elements:
        if el.category == "Table":
            table_html = el.metadata.text_as_html
            final_text += table_html
        else:
            new_text = el.text
            final_text += new_text
    
    with open("RFP_requirements.txt", "w") as f:
        f.write(final_text)
        
    with st.expander("text output"):
        st.write(final_text)

    return resp, elements, tables, final_text



#
# Main
#

st.markdown("# Upload RFP File in PDF Format")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")
if uploaded_file is not None:
    file_contents = uploaded_file.getbuffer()
    file_name = uploaded_file.name
    resp, elements, tables, final_text = process_file(file_contents, file_name)
    st.write("Processing complete")




