import streamlit as st

# Title
st.title('Medical Report Analyzer')

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report', type=['pdf', 'docx', 'txt'])

# Process file
if uploaded_file is not None:
    # Display file details
    st.write('File name:', uploaded_file.name)
    content = uploaded_file.read()
    st.write('File content:', content.decode('utf-8') if isinstance(content, bytes) else content)
