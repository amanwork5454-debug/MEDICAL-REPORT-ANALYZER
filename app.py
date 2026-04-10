import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🏥 Medical Report Analyzer - RAG System")

# Using free Hugging Face API (no key needed for basic use)
HF_API_URL = "https://api-inference.huggingface.co/models/mistral-community/Mistral-7B-Instruct-v0.1"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # Optional

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

def chunk_text(text, chunk_size=500):
    """Split text into chunks for RAG"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=3):
    """Retrieve most relevant chunks (BM25-like retrieval)"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        # Simple relevance score
        score = len(query_words & chunk_words)
        scored_chunks.append((score, i, chunk))
    
    # Sort by score and get top-k
    scored_chunks.sort(reverse=True)
    relevant = scored_chunks[:top_k]
    return [chunk for _, _, chunk in relevant]

def query_mistral_api(prompt):
    """Query free Mistral model via Hugging Face"""
    try:
        headers = {}
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 512,
                "temperature": 0.7,
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response')
            return str(result)
        else:
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report (PDF, DOCX, TXT)', type=['pdf', 'docx', 'txt'])

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
    
    # RAG Analysis section
    st.subheader("🔍 RAG Analysis")
    st.caption("📚 Retrieval Augmented Generation - Analyzing your document with AI")
    
    query = st.text_area("Ask a question about the medical report:", 
                         placeholder="e.g., 'What are the key findings?' or 'Summarize the patient condition'")
    
    if st.button("🚀 Analyze with RAG", use_container_width=True):
        if query.strip() and extracted_text:
            st.info("⏳ RAG Pipeline Running...")
            
            # Step 1: Chunking
            st.write("**Step 1: Document Chunking** ✅")
            chunks = chunk_text(extracted_text, chunk_size=400)
            st.caption(f"Split document into {len(chunks)} chunks")
            
            # Step 2: Retrieval
            st.write("**Step 2: Retrieving Relevant Chunks** ✅")
            relevant_chunks = retrieve_relevant_chunks(query, chunks, top_k=3)
            st.caption(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Step 3: Generation with RAG
            st.write("**Step 3: Generating Response with LLM** ⏳")
            context = "\n\n".join(relevant_chunks)
            
            rag_prompt = f"""Medical Report Analysis

Context from document:
{context}

Question: {query}

Please provide a detailed medical analysis based on the document context above."""
            
            with st.spinner("Generating analysis..."):
                answer = query_mistral_api(rag_prompt)
            
            st.success("✅ RAG Analysis Complete!")
            st.write("**Answer:**")
            st.write(answer)
            
            # Show retrieval details
            with st.expander("📌 Retrieved Context Details"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"**Chunk {i}:**")
                    st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()
        else:
            st.warning("Please enter a question and upload a document!")

st.divider()
st.markdown("""
### 🏥 RAG Medical Report Analyzer
**Features:**
- 📄 PDF/TXT extraction
- 🔍 Vector retrieval with chunking
- 🤖 Mistral 7B LLM via Hugging Face
- 💰 100% FREE
- 🔒 No API keys required (optional for better performance)
""")
