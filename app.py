import os
import gradio as gr

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import heapq

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

vectorstore = None


def load_document(file_path):
    ext = os.path.splitext(file_path)[1]

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        return None

    return loader.load()


def process_file(file):
    global vectorstore

    if file is None:
        return "Upload a file first"

    docs = load_document(file.name)
    if docs is None:
        return "Unsupported file type"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return "File processed successfully"


def ask_question(question):
    global vectorstore

    if vectorstore is None:
        return "Upload and process a file first"

    llm = ChatGroq(model="openai/gpt-oss-120b")

    all_docs = vectorstore.docstore._dict.values()
    context_text = " ".join([d.page_content for d in all_docs])

    response = llm.invoke(f"Answer only from this context:\n{context_text}\nQuestion:{question}")

    source_text = "\n\nSources:\n"
    for idx, d in enumerate(all_docs):
        page = d.metadata.get("page", "N/A")
        source_text += f"- Document {idx+1}, Page: {page}\n"

    return response.content + source_text


def evaluate_model(top_k=5):
    global vectorstore

    if vectorstore is None:
        return "Process a file first"

    llm = ChatGroq(model="openai/gpt-oss-120b")
    ragas_llm = LangchainLLMWrapper(llm)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    questions = ["What are the payment terms?"]
    ground_truths = ["termination clause", "payment terms", "contract duration"]

    input_texts = []
    predicted_answers = []
    context_lists = []
    gt_list = []

    all_docs = list(vectorstore.docstore._dict.values())

    for q in questions:
       
        top_chunks = all_docs[:top_k]
        context_text = " ".join([d.page_content for d in top_chunks])
        response = llm.invoke(f"Answer only from this context:\n{context_text}\nQuestion:{q}")

        for gt in ground_truths:
            input_texts.append(q)
            predicted_answers.append(response.content)
            context_lists.append([d.page_content for d in top_chunks])
            gt_list.append(gt)

    data = {
        "question": input_texts,
        "answer": predicted_answers,
        "contexts": context_lists,
        "ground_truth": gt_list
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return str(result)


with gr.Blocks() as demo:
    gr.Markdown("Smart Contract Assistant")

    file_input = gr.File(file_types=[".pdf", ".txt", ".docx", ".csv", ".pptx"])
    process_btn = gr.Button("Process File")
    process_output = gr.Textbox()

    question = gr.Textbox(label="Ask a Question")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox()

    eval_btn = gr.Button("Run Evaluation")
    eval_output = gr.Textbox(label="Evaluation Results")

    process_btn.click(process_file, file_input, process_output)
    ask_btn.click(ask_question, question, answer)
    eval_btn.click(evaluate_model, outputs=eval_output)

demo.launch()