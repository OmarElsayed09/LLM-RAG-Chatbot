# Repeatable retrieval checks and real demo capture

## Retrieval evaluation

Run after installing a Python environment with `langchain-core`, `langchain-community`, `langchain-huggingface`, `faiss-cpu` and `sentence-transformers`:

```bash
python evaluate_retrieval.py --dataset eval/sample_questions.json --top-k 1 --output artifacts/retrieval-metrics.json
```

The generated JSON includes corpus SHA256, Git commit, Python/model package version, per-question retrieved passage IDs, hit@k and recall@k. **No scores are hard-coded.** The included data are synthetic, and any score on them is a *smoke-test result*, not validated production accuracy.

To evaluate your own research/document collections, replace the JSON with a reviewed passage set and question-to-relevant-passage labels. Specify relevant answers and measure generated-answer faithfulness separately; this script measures retrieval only.

GitHub Actions workflow `retrieval-evaluation.yml` runs this experiment on pull requests and can upload the actual generated report.

## Demo screenshot

The manually triggered `demo-capture.yml` workflow boots the actual Gradio app and uses a headless Chromium browser to capture the homepage as an **Actions artifact**. This screenshot is real UI output, not an AI-generated mockup. The demo homepage does **not** prove successful upload, indexing, or answer generation.

A verified *end-to-end* demo additionally requires a test file, access to embeddings, `GROQ_API_KEY` for generation, manual removal of sensitive inputs, and inspection of responses. Do not publish API keys or private documents.
