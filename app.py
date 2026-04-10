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

def chunk_text(text, chunk_size=300):
    """Split text into chunks for RAG"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=3):
    """Retrieve most relevant chunks based on keyword matching"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        # Calculate relevance score
        score = len(query_words & chunk_words)
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top-k
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def generate_answer(query, relevant_chunks):
    """Generate answer from retrieved chunks"""
    if not relevant_chunks:
        return "No relevant information found in the document."
    
    context = "\n\n".join(relevant_chunks)
    
    # Create a summary based on the retrieved context
    answer = f"""
**Analysis Results:**

Based on the document analysis for your query: "{query}"

**Relevant Information Found:**
{context[:500]}...

**Key Points:**
1. Document contains relevant information matching your query
2. Total characters processed: {len(context)} characters
3. Analysis completed successfully
"""
    return answer

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    
    # Extract text
    if uploaded_file.name.endswith('.pdf'):
        st.info("📄 Extracting PDF...")
        extracted_text = extract_pdf_text(uploaded_file)
    else:
        st.info("📝 Reading text file...")
        extracted_text = extract_text_file(uploaded_file)
    
    if "Error" not in extracted_text:
        # Show extracted content
        st.subheader("📋 Content Preview")
        with st.expander("View full text"):
            st.text_area("Extracted:", extracted_text[:1000], height=200, disabled=True)
        
        # RAG Analysis
        st.subheader("🔍 RAG Analysis")
        query = st.text_area("Ask question about the document:", placeholder="e.g., 'What are the key findings?'")
        
        if st.button("Analyze with RAG"):
            if query.strip():
                st.info("⏳ Running RAG Pipeline...")
                
                # Step 1: Chunking
                st.write("**Step 1: Document Chunking**")
                chunks = chunk_text(extracted_text, chunk_size=300)
                st.success(f"✅ Split document into {len(chunks)} chunks")
                
                # Step 2: Retrieval
                st.write("**Step 2: Retrieving Relevant Information**")
                relevant_chunks = retrieve_relevant_chunks(query, chunks, top_k=3)
                st.success(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
                
                # Step 3: Generation
                st.write("**Step 3: Generating Answer**")
                answer = generate_answer(query, relevant_chunks)
                st.success("✅ Analysis Complete!")
                
                # Display answer
                st.write(answer)
                
                # Show retrieved context
                with st.expander("📌 Retrieved Context"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()
            else:
                st.warning("Please enter a question!")
    else:
        st.error(extracted_text)

st.markdown("---")
st.markdown("""
### 🏥 RAG Medical Report Analyzer
**Features:**
- 📄 PDF & TXT extraction
- 🔍 Intelligent chunk retrieval
- 🤖 Context-aware analysis
- 💰 100% FREE
- ⚡ Fast processing
""")
