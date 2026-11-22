"""Repeatable FAISS retrieval evaluation with locally computed embeddings.

This example fixture is synthetic and measures pipeline wiring, not real-world RAG
performance. Supply your own labeled JSON to assess actual document retrieval.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def evaluate(dataset_path, top_k=1):
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import sentence_transformers

    source = Path(dataset_path)
    raw = source.read_bytes()
    dataset = json.loads(raw)
    passages = dataset["passages"]
    questions = dataset["questions"]
    if not passages or not questions or top_k < 1:
        raise ValueError("Provide passages, questions, and a positive top_k.")
    ids = [p["id"] for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("Passage IDs must be unique.")
    docs = [Document(page_content=p["text"], metadata={"passage_id": p["id"]}) for p in passages]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    store = FAISS.from_documents(docs, embeddings)
    rows = []
    for case in questions:
        expected = set(case["relevant_passage_ids"])
        if not expected or not expected.issubset(set(ids)):
            raise ValueError("All questions must reference known relevant passage IDs.")
        found = [d.metadata["passage_id"] for d in store.similarity_search(case["question"], k=top_k)]
        rows.append({
            "question": case["question"],
            "expected_passage_ids": sorted(expected),
            "retrieved_passage_ids": found,
            "hit": bool(expected.intersection(found)),
            "recall": len(expected.intersection(found)) / len(expected),
        })
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_description": dataset.get("description", ""),
        "embedding_model": model_name,
        "sentence_transformers_version": sentence_transformers.__version__,
        "python_version": platform.python_version(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None,
        "top_k": top_k,
        "question_count": len(rows),
        "hit_rate_at_k": sum(r["hit"] for r in rows) / len(rows),
        "mean_recall_at_k": sum(r["recall"] for r in rows) / len(rows),
        "results": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval over a labeled dataset")
    parser.add_argument("--dataset", default="eval/sample_questions.json")
    parser.add_argument("--output", default="artifacts/retrieval-metrics.json")
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    report = evaluate(args.dataset, args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}: hit@{args.top_k}={report['hit_rate_at_k']:.3f}; recall@{args.top_k}={report['mean_recall_at_k']:.3f}")
