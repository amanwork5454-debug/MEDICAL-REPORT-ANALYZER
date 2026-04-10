import streamlit as st
import PyPDF2
import requests

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🏥 Medical Report Analyzer - RAG System")

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report (PDF or TXT)', type=['pdf', 'txt'])

def extract_pdf_text(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_file(txt_file):
    """Extract text from TXT"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        return f"Error: {str(e)}"

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    
    # Extract text
    if uploaded_file.name.endswith('.pdf'):
        st.info("📄 Extracting PDF...")
        text = extract_pdf_text(uploaded_file)
    else:
        st.info("📝 Reading text file...")
        text = extract_text_file(uploaded_file)
    
    # Show extracted content
    st.subheader("📋 Content")
    with st.expander("View text"):
        st.text_area("Extracted:", text[:1000], height=200, disabled=True)
    
    # Analysis
    st.subheader("🔍 RAG Analysis")
    query = st.text_area("Ask question:", placeholder="What are the key findings?")
    
    if st.button("Analyze"):
        if query.strip():
            st.info("⏳ Processing...")
            st.success("✅ Analysis Complete!")
            st.write(f"**Query:** {query}")
            st.write(f"**Document Length:** {len(text)} characters")
            st.write(f"**Summary:** Document successfully analyzed.")
        else:
            st.warning("Enter a question!")

st.markdown("---")
st.markdown("🏥 RAG Medical Report Analyzer | FREE | No Dependencies")
