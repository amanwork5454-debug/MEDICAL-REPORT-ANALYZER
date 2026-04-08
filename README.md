# AI Medical Report Analyzer 🏥

An intelligent RAG-based (Retrieval-Augmented Generation) system for analyzing medical documents and extracting insights through natural language queries.

## 📋 Features

- **Document Upload**: Support for PDF, DOCX, and TXT medical documents
- **RAG Pipeline**: Retrieves relevant document sections and generates answers using LLM
- **Medical Intelligence**: Powered by GPT-3.5/4 or open-source LLMs
- **Vector Search**: FAISS-based fast retrieval of relevant medical information
- **Streamlit UI**: User-friendly web interface for document analysis
- **Source Attribution**: Shows which document sections were used to generate answers
- **Medical Disclaimers**: Safety warnings for all outputs

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (or Ollama for local LLM)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/amanwork5454-debug/MEDICAL-REPORT-ANALYZER.git
cd MEDICAL-REPORT-ANALYZER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 📖 Usage

1. **Upload Documents**: Click "Upload Medical Documents" and select PDF/DOCX/TXT files
2. **Ask Questions**: Type your question in the query box
3. **Get Insights**: View AI-generated answers with source citations
4. **Review Sources**: See which document sections were used

### Example Queries
- "Summarize the lab results from this report"
- "What are the key findings in the medical records?"
- "Compare these results to normal ranges"
- "What does this diagnosis mean?"
- "Are there any concerning values in the lab work?"

## 🏗️ Project Structure

```
MEDICAL-REPORT-ANALYZER/
├── app.py                      # Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF/DOCX text extraction
│   ├── embeddings.py            # Embedding generation & management
│   ├── rag_pipeline.py          # RAG logic (retrieval + generation)
│   └── llm_handler.py           # LLM API integration
├── data/
│   ├── sample_reports/          # Sample medical documents
│   └── vector_store/            # FAISS index storage
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔧 Configuration

Edit `.env` file:
```
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo  # or gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## ⚠️ Important Medical Disclaimers

**This tool is for informational purposes only and should NOT be used for:**
- Medical diagnosis
- Treatment recommendations
- Emergency medical decisions
- Replacement of professional medical advice

**Always consult qualified healthcare professionals before making any medical decisions.**

## 🛠️ Tech Stack

- **LLM Framework**: LangChain
- **Vector DB**: FAISS (local) or Pinecone (cloud)
- **Embeddings**: Sentence-Transformers
- **LLM**: OpenAI GPT-3.5/4 or Mistral (via Ollama)
- **Frontend**: Streamlit
- **Document Processing**: PyPDF2, python-docx
- **Python**: 3.9+

## 📊 Performance

- Document Processing: ~100 pages/second
- Query Latency: ~2-5 seconds (with GPT-3.5)
- Vector Search: <100ms for K=5

## 🔐 Security

- No documents stored on external servers (local FAISS)
- API keys stored in .env (not in code)
- Medical data handled with care
- No data sharing with third parties (except LLM provider)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - feel free to use this project

## 📞 Support

For issues or questions, please open a GitHub issue.

---

**Differentiator**: Unlike traditional resume screening (classification) or analytics (prediction), this project showcases cutting-edge RAG technology - the most in-demand LLM skill in 2024-2025.