import streamlit as st
import PyPDF2
import re
import requests
import json

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🏥 Medical Report Analyzer - RAG Gen AI")

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")

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

def chunk_text(text, chunk_size=200):
    """Split text into chunks for RAG"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_size += len(sentence.split())
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c.strip() for c in chunks if c.strip()]

def retrieve_relevant_chunks(query, chunks, top_k=5):
    """Retrieve most relevant chunks with fuzzy matching"""
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Check if query appears as substring (exact match)
        if query_lower in chunk_lower:
            score = 1000  # Very high score for exact match
        else:
            # Count matching words
            score = 0
            for word in query_words:
                if len(word) > 2:  # Only count words longer than 2 chars
                    if word in chunk_lower:
                        score += 10
                    # Check for partial matches
                    for chunk_word in chunk_lower.split():
                        if word in chunk_word or chunk_word in word:
                            score += 2
        
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top-k
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def generate_answer_with_llama(query, relevant_chunks):
    """Generate answer using Llama 2 Gen AI model"""
    
    if not relevant_chunks:
        return "No relevant information found in the document."
    
    # Create context from retrieved chunks
    context = "\n".join(relevant_chunks)
    
    # Create prompt for Llama 2
    prompt = f"""[INST] You are a helpful medical report analyzer. Based on the following document context, answer the user's question clearly and concisely.

Document Context:
{context}

User Question: {query}

Please provide a clear and accurate answer based only on the provided context. If the information is not in the context, say so. [/INST]"""
    
    try:
        # Call Hugging Face API
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Extract the answer part (after [/INST])
                if '[/INST]' in generated_text:
                    answer = generated_text.split('[/INST]')[-1].strip()
                else:
                    answer = generated_text
                return answer
            return "Unable to generate answer"
        else:
            return f"API Error: {response.status_code}. Make sure you have a valid Hugging Face API token in Streamlit Secrets."
            
    except requests.exceptions.Timeout:
        return "Request timed out. The model is taking too long to respond. Please try again."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# File uploader
uploaded_file = st.file_uploader('Upload Medical Report (PDF or TXT)', type=['pdf', 'txt'])

if uploaded_file is not None:
    st.write(f"**📎 File:** {uploaded_file.name}")
    st.write(f"**📏 Size:** {uploaded_file.size / 1024:.2f} KB")
    
    # Extract text
    if uploaded_file.name.endswith('.pdf'):
        with st.spinner("📄 Extracting PDF..."):
            extracted_text = extract_pdf_text(uploaded_file)
    else:
        with st.spinner("📝 Reading text file..."):
            extracted_text = extract_text_file(uploaded_file)
    
    if "Error" not in extracted_text and len(extracted_text) > 10:
        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", len(extracted_text.split()))
        with col2:
            st.metric("Characters", len(extracted_text))
        with col3:
            st.metric("Paragraphs", len(extracted_text.split('\n\n')))
        
        # Show preview
        with st.expander("👁️ View Document Preview"):
            st.text_area("Content:", extracted_text[:1500], height=200, disabled=True)
        
        # RAG Gen AI Analysis
        st.subheader("🤖 Ask Questions (Powered by Llama 2 Gen AI)")
        st.caption("⚡ This uses Llama 2 LLM to generate intelligent answers from your document")
        
        query = st.text_input("Enter your question:", placeholder="e.g., 'What is the name?', 'What are the key findings?'")
        
        if st.button("🚀 Analyze with RAG Gen AI", use_container_width=True):
            if query.strip():
                st.info("⏳ Running RAG Gen AI Pipeline...")
                
                # Step 1: Chunking
                with st.spinner("**Step 1:** Chunking document..."):
                    chunks = chunk_text(extracted_text, chunk_size=200)
                st.success(f"✅ **Step 1 Complete:** Split into {len(chunks)} chunks")
                
                # Step 2: Retrieval
                with st.spinner("**Step 2:** Retrieving relevant information..."):
                    relevant_chunks = retrieve_relevant_chunks(query, chunks, top_k=5)
                st.success(f"✅ **Step 2 Complete:** Retrieved {len(relevant_chunks)} relevant sections")
                
                # Step 3: Generation with Llama 2
                with st.spinner("**Step 3:** Generating answer with Llama 2 LLM..."):
                    answer = generate_answer_with_llama(query, relevant_chunks)
                st.success("✅ **Step 3 Complete:** Answer generated!")
                
                # Display answer
                st.info(f"**Query:** {query}")
                st.write(f"**Answer:**\n\n{answer}")
                
                # Show retrieved context
                with st.expander("📌 View Retrieved Context Used"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(chunk)
                        st.divider()
            else:
                st.warning("⚠️ Please enter a question!")
    else:
        st.error("❌ Failed to extract text from file. Please try another file.")

st.markdown("---")
st.markdown("""
### 🏥 RAG Gen AI Medical Report Analyzer

**Architecture:**
1. 📄 **Document Input** - Extract text from PDF/TXT
2. ✂️ **Chunking** - Split into manageable chunks
3. 🔍 **Retrieval** - Find most relevant chunks (BM25-like)
4. 🤖 **Generation** - Llama 2 LLM generates intelligent answers

**Model:** Meta Llama 2 7B Chat (via Hugging Face)

**Features:**
- ✅ True RAG Gen AI system
- ✅ LLM-powered answers
- ✅ Document context-aware
- ✅ Fast inference
- ✅ 100% FREE

**Setup Required:**
1. Get free Hugging Face token: https://huggingface.co/settings/tokens
2. Add to Streamlit Secrets: `HF_API_TOKEN`
""")
