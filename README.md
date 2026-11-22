# Smart Document Assistant — RAG Prototype

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Quality checks](https://github.com/OmarElsayed09/LLM-RAG-Chatbot/actions/workflows/python-checks.yml/badge.svg)](https://github.com/OmarElsayed09/LLM-RAG-Chatbot/actions/workflows/python-checks.yml)

A Gradio application for asking grounded questions about uploaded PDF, TXT, DOCX, CSV, and PPTX files. Text is chunked, embedded with a local Sentence Transformers model, indexed in FAISS, and retrieved by semantic similarity before a Groq-hosted LLM produces an answer.

> **Status: learning / portfolio prototype.** No production reliability or answer-quality guarantees are claimed. Outputs can be incorrect even with retrieval.

## How it works

```text
Upload document → extract text → split chunks → embed → FAISS index
Question → similarity search (top 4 chunks) → Groq LLM → answer + retrieved sources
```

## Local setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your shell or load it from a local `.env` via your preferred method. See `.env.example`; **never commit the real key**.

```bash
python app.py
```

Open the local Gradio URL, upload one file, click **Process file**, and ask a question. The index is held in process memory and overwritten on the next upload.

## Stack

Python, Gradio, LangChain, Hugging Face embeddings, FAISS, Groq.

## Evaluation and limitations

The old evaluation button used unrelated placeholder ground truths and was **not a valid benchmark**. It now explicitly reports that labeled evaluation data is required. To evaluate rigorously, create document-specific questions with human-reviewed reference answers, measure retrieval recall and groundedness, and record dataset versions.

- API access and internet are required for Groq generation.
- PPTX extraction via Unstructured may require additional system dependencies.
- Do not upload confidential or regulated material to an external model without authorization.
- Retrieved chunks are listed as sources, not verified claim-level citations.
- This repository is not a production service.

## Author

[Omar Elsayed](https://github.com/OmarElsayed09) · [Portfolio](https://omar-elsayed-portfolio.vercel.app/)
