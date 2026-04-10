import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🏥 Medical Report Analyzer")

# Get API key
api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API key not found! Add it to Streamlit Secrets.")
    st.stop()

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
            st.info("⏳ Analyzing with OpenAI...")
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a medical document analyzer. Provide clear, professional analysis."},
                        {"role": "user", "content": f"Document content:\n{extracted_text[:2000]}\n\nQuestion: {query}"}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content
                st.success("✅ Analysis Complete!")
                st.write(answer)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question!")

st.divider()
st.markdown("🏥 Medical Report Analyzer | Powered by OpenAI")
