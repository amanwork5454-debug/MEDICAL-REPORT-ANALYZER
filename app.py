import streamlit as st
import PyPDF2
import os

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🏥 Medical Report Analyzer")

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report (PDF, DOCX, TXT)', type=['pdf', 'docx', 'txt'])

def extract_pdf_text(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_file(txt_file):
    """Extract text from TXT"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

if uploaded_file is not None:
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
    
    # Extract text based on file type
    if uploaded_file.name.endswith('.pdf'):
        st.info("📄 PDF detected. Extracting text...")
        extracted_text = extract_pdf_text(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        st.info("📝 TXT file detected. Reading...")
        extracted_text = extract_text_file(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        st.warning("📎 DOCX support coming soon!")
        extracted_text = "DOCX processing not yet implemented"
    else:
        extracted_text = ""
    
    # Display extracted content
    st.subheader("📋 Extracted Content:")
    with st.expander("View full text", expanded=False):
        st.text_area("Content:", extracted_text, height=300, disabled=True)
    
    # Analysis section
    st.subheader("🔍 Analysis")
    query = st.text_area("Ask a question about the document:", 
                         placeholder="e.g., 'Summarize the findings' or 'What are the test results?'")
    
    if st.button("Analyze with AI", use_container_width=True):
        if query.strip():
            st.info("⏳ Analyzing document using free AI model...")
            try:
                from transformers import pipeline
                
                # Load free summarization model
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                
                # Summarize the extracted text
                if len(extracted_text) > 100:
                    summary = summarizer(extracted_text[:1024], max_length=150, min_length=50, do_sample=False)
                    result = summary[0]['summary_text']
                    
                    st.success("✅ Analysis Complete!")
                    st.write("**Summary:**")
                    st.write(result)
                else:
                    st.warning("Document too short to summarize")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question!")

st.divider()
st.markdown("🏥 Medical Report Analyzer | FREE AI Powered | No API Cost Required ✨")
