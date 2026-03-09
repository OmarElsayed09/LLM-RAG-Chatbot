# Smart Document Assistant (RAG Application)

## Project Description

This project is a Retrieval-Augmented Generation (RAG) based Smart Document Assistant that allows users to upload different document formats and ask questions about their content.

The system processes uploaded files, extracts the text, converts it into embeddings, and uses an LLM to generate contextual answers based on the document content.

---

## Features

- Upload multiple document formats:
  - PDF
  - TXT
  - DOCX
  - CSV
  - PPTX
- Automatic document processing
- Semantic search using vector database
- Context-aware question answering
- Free LLM using Groq API
- Interactive UI using Gradio

---

## Technologies Used

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Gradio

---

## Installation

### 1. Create virtual environment

```bash
python -m venv venv