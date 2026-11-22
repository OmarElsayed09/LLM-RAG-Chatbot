"""Gradio document question-answering prototype with retrieved context."""
import os
from pathlib import Path

import gradio as gr
from langchain_community.document_loaders import (
    CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredPowerPointLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

vectorstore = None
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def load_document(path):
    extension = Path(path).suffix.lower()
    if extension not in LOADERS:
        raise ValueError(f"Unsupported file type: {extension}")
    return LOADERS[extension](str(path)).load()


def retrieve_context(store, question, k=4):
    """Fetch relevant chunks instead of sending the entire document to the LLM."""
    if not question or not question.strip():
        return []
    return store.similarity_search(question, k=k)


def process_file(file):
    global vectorstore
    if file is None:
        return "Upload a file first."
    try:
        docs = load_document(file.name)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        if not chunks:
            return "The file contained no extractable text."
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return f"Indexed {len(chunks)} text chunks."
    except Exception as exc:
        return f"Could not process file: {exc}"


def ask_question(question):
    if vectorstore is None:
        return "Upload and process a document first."
    if not question or not question.strip():
        return "Enter a question."
    if not os.getenv("GROQ_API_KEY"):
        return "Set GROQ_API_KEY in your environment before asking questions."
    docs = retrieve_context(vectorstore, question)
    if not docs:
        return "No matching document passages were found."
    context = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))
    prompt = (
        "Answer only using the context below. If the information is absent, "
        "say you cannot find it in the uploaded document. Do not invent facts.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    try:
        answer = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        ).invoke(prompt).content
    except Exception as exc:
        return f"LLM request failed: {exc}"
    sources = []
    for index, doc in enumerate(docs, 1):
        name = Path(str(doc.metadata.get("source", "uploaded document"))).name
        page = doc.metadata.get("page")
        location = f", page {page + 1}" if isinstance(page, int) else ""
        sources.append(f"[{index}] {name}{location}")
    return f"{answer}\n\nRetrieved sources:\n" + "\n".join(sources)


def evaluate_model():
    return (
        "Evaluation requires a representative, labeled question-answer dataset. "
        "No benchmark score is reported from placeholder questions."
    )


with gr.Blocks(title="Smart Document Assistant") as demo:
    gr.Markdown("# Smart Document Assistant\nQuestion answering over uploaded documents (prototype).")
    file_input = gr.File(file_types=list(LOADERS))
    process_btn = gr.Button("Process file")
    process_output = gr.Textbox(label="Indexing status")
    question = gr.Textbox(label="Ask a question")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer and retrieved sources", lines=12)
    eval_btn = gr.Button("Evaluation status")
    eval_output = gr.Textbox(label="Evaluation")
    process_btn.click(process_file, inputs=file_input, outputs=process_output)
    ask_btn.click(ask_question, inputs=question, outputs=answer)
    eval_btn.click(evaluate_model, outputs=eval_output)


if __name__ == "__main__":
    demo.launch()
