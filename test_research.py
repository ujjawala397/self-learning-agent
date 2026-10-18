import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import research


class ResearchTests(unittest.TestCase):
    def test_research_returns_no_evidence_without_calling_answer_model(self) -> None:
        with (
            patch("research.retrieve_passages", return_value=[]) as retrieve,
            patch("research.OpenAI") as openai,
        ):
            result = research.research("Question about Reliance")

        self.assertEqual(
            result,
            {"answer": research.NO_EVIDENCE_ANSWER, "sources": []},
        )
        retrieve.assert_called_once_with(
            question="Question about Reliance",
            top_k=8,
            financial_year=None,
        )
        openai.assert_not_called()

    def test_research_generates_answer_only_from_labeled_evidence(self) -> None:
        passage = {
            "source_file": "RIL-FY2024-25.pdf",
            "financial_year": "FY2024-25",
            "page_number": 56,
            "text": "The report says revenue was 100 units.",
        }
        model = Mock()
        model.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Revenue was 100 [S1]."))
        ]
        with (
            patch(
                "research.retrieve_passages", return_value=[passage]
            ) as retrieve,
            patch("research.OpenAI", return_value=model),
        ):
            result = research.research(
                "What was revenue?", financial_year="FY2024-25", top_k=3
            )

        self.assertEqual(result["answer"], "Revenue was 100 [S1].")
        self.assertEqual(
            result["sources"],
            [
                {
                    "id": "S1",
                    "source_file": "RIL-FY2024-25.pdf",
                    "financial_year": "FY2024-25",
                    "page_number": 56,
                    "text": "The report says revenue was 100 units.",
                }
            ],
        )
        retrieve.assert_called_once_with(
            question="What was revenue?",
            top_k=3,
            financial_year="FY2024-25",
        )
        prompt = model.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("using only the supplied", prompt[0]["content"])
        self.assertIn("[S1]", prompt[1]["content"])
        self.assertIn("RIL-FY2024-25.pdf", prompt[1]["content"])
        self.assertIn("PDF page: 56", prompt[1]["content"])

    def test_empty_question_and_invalid_top_k_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            research.research("  ")
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            research.research("Question", top_k=0)

    def test_empty_model_answer_uses_explicit_fallback(self) -> None:
        model = Mock()
        model.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content=None))
        ]
        with (
            patch("research.retrieve_passages", return_value=[{"text": "evidence"}]),
            patch("research.OpenAI", return_value=model),
        ):
            result = research.research("Question")

        self.assertEqual(result["answer"], research.NO_ANSWER)
        self.assertEqual(result["sources"][0]["id"], "S1")

    def test_stream_research_emits_sources_before_cited_answer_tokens(self) -> None:
        passage = {
            "source_file": "RIL-FY2024-25.pdf",
            "financial_year": "FY2024-25",
            "page_number": 56,
            "text": "Annual revenue is 100 units.",
        }
        model = Mock()
        model.chat.completions.create.return_value = iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="100 "))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="[S1]."))]
                ),
            ]
        )
        with (
            patch("research.retrieve_passages", return_value=[passage]),
            patch("research.OpenAI", return_value=model),
        ):
            events = list(research.stream_research("What was annual revenue?"))

        self.assertEqual(
            events,
            [
                {
                    "type": "sources",
                    "sources": [
                        {
                            "id": "S1",
                            "source_file": "RIL-FY2024-25.pdf",
                            "financial_year": "FY2024-25",
                            "page_number": 56,
                        }
                    ],
                },
                {"type": "token", "token": "100 "},
                {"type": "token", "token": "[S1]."},
                {"type": "done"},
            ],
        )
        call = model.chat.completions.create.call_args.kwargs
        self.assertTrue(call["stream"])
        self.assertIn("annual-report evidence", call["messages"][1]["content"])

    def test_stream_research_with_no_sources_skips_llm(self) -> None:
        with (
            patch("research.retrieve_passages", return_value=[]),
            patch("research.OpenAI") as openai,
        ):
            events = list(research.stream_research("Question"))

        self.assertEqual(
            events,
            [
                {"type": "sources", "sources": []},
                {"type": "token", "token": research.NO_EVIDENCE_ANSWER},
                {"type": "done"},
            ],
        )
        openai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
