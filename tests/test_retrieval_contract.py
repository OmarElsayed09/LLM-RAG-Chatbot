"""Fast structural test; no API key or heavyweight downloads required."""
from pathlib import Path
import ast


def test_app_defines_similarity_retrieval():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert "retrieve_context" in functions
    assert "ask_question" in functions
    assert "similarity_search" in source
    assert 'os.environ["GROQ_API_KEY"]' not in source
