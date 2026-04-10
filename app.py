import streamlit as st

st.title("Medical Report Analyzer")

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report', type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    st.write('File name:', uploaded_file.name)
    
    # Handle different file types
    if uploaded_file.name.endswith('.pdf'):
        st.info("PDF uploaded. Processing...")
        # PDF processing logic here
    elif uploaded_file.name.endswith('.docx'):
        st.info("DOCX uploaded. Processing...")
        # DOCX processing logic here
    elif uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode('utf-8')
        st.write('File content:', content)
    else:
        st.error("Unsupported file type")
