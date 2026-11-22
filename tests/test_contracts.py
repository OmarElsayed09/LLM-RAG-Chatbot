"""Dependency-free tests of retrieval contracts and benchmark honesty."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RAGContracts(unittest.TestCase):
    def test_similarity_search_is_used(self):
        text = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("retrieve_context", functions)
        self.assertIn("ask_question", functions)
        self.assertIn("similarity_search", text)
        self.assertNotIn('os.environ["GROQ_API_KEY"]', text)

    def test_requirements_are_one_per_line(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        packages = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertTrue(packages)
        self.assertTrue(all(" " not in package for package in packages))

    def test_evaluation_does_not_report_dummy_scores(self):
        text = (ROOT / "evaluation.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("similarity_search", text)
        self.assertNotIn('ground_truths = ["termination clause"', text)


if __name__ == "__main__":
    unittest.main()
