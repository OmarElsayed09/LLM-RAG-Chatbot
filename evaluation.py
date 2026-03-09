from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import heapq

def run_evaluation(vectorstore, top_k=5):
    """
    Evaluation fast                                 
    """
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

        scored_chunks = []
        for d in all_docs:
     
            if hasattr(d, "embedding") and d.embedding is not None:
  
                similarity = vectorstore.similarity_search_with_score(q, k=1)

                similarity_score = similarity[0][1] if similarity else 0
            else:
                similarity_score = 0
            scored_chunks.append((similarity_score, d))

        top_chunks = [d for score, d in heapq.nlargest(top_k, scored_chunks, key=lambda x: x[0])]

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

    return result