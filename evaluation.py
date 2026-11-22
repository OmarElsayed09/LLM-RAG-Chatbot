"""Utilities for inspecting retrieval before building a labeled benchmark.

No quality metric is computed without a verified evaluation dataset.
"""


def retrieve_for_review(vectorstore, questions, top_k=5):
    """Return question/chunk pairs for a human to inspect and label.

    Args:
        vectorstore: LangChain vector store implementing similarity_search.
        questions: Collection of real questions about the indexed documents.
        top_k: Maximum number of retrieved chunks per question.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reviews = []
    for question in questions:
        if not str(question).strip():
            continue
        chunks = vectorstore.similarity_search(question, k=top_k)
        reviews.append({
            "question": question,
            "retrieved_contexts": [chunk.page_content for chunk in chunks],
            "metadata": [chunk.metadata for chunk in chunks],
        })
    return reviews
