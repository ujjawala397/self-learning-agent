import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
import web_app


class FakeMemory:
    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []

    def add(self, **kwargs: object) -> None:
        self.added.append(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=" there."))]
                ),
            ]
        )


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.memory = FakeMemory()
        self.app = web_app.create_app(self.client, self.memory)
        self.app.testing = True
        self.http = self.app.test_client()

    def test_chat_streams_tokens_and_records_user_message_after_completion(self) -> None:
        with (
            patch("agent.get_memory_context", return_value=""),
            patch("agent.get_feedback_context", return_value=""),
        ):
            response = self.http.post(
                "/api/chat",
                json={"question": "Say hello"},
                buffered=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        events = [
            json.loads(line[6:])
            for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(
            events,
            [{"token": "Hello"}, {"token": " there."}, {"done": True}],
        )
        self.assertTrue(self.client.calls[0]["stream"])
        self.assertEqual(
            self.memory.added,
            [
                {
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "user_id": agent.USER_ID,
                    "prompt": agent.MEMORY_POLICY,
                }
            ],
        )

    def test_chat_rejects_empty_question(self) -> None:
        response = self.http.post("/api/chat", json={"question": " "})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Enter a question", response.get_json()["error"])

    def test_feedback_api_saves_user_confirmed_correction(self) -> None:
        with patch("agent.record_feedback", return_value=15) as save_feedback:
            response = self.http.post(
                "/api/feedback",
                json={
                    "question": "What is 1 + 1?",
                    "answer": "3",
                    "rating": "bad",
                    "explanation": "The arithmetic was wrong.",
                    "correction": "1 + 1 = 2.",
                    "confirmed": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"feedback_id": 15})
        save_feedback.assert_called_once_with(
            "What is 1 + 1?",
            "3",
            "bad",
            explanation="The arithmetic was wrong.",
            correction="1 + 1 = 2.",
            verified=True,
        )

    def test_positive_rating_is_not_marked_verified(self) -> None:
        with patch("agent.record_feedback", return_value=16) as save_feedback:
            response = self.http.post(
                "/api/feedback",
                json={
                    "question": "Question",
                    "answer": "Answer",
                    "rating": "good",
                    "confirmed": True,
                    "correction": "not applicable",
                },
            )

        self.assertEqual(response.status_code, 200)
        save_feedback.assert_called_once_with(
            "Question",
            "Answer",
            "good",
            explanation=None,
            correction=None,
            verified=False,
        )


if __name__ == "__main__":
    unittest.main()
