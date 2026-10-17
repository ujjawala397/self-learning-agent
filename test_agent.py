import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent


class FakeMemory:
    def __init__(self, owner_id: str = agent.USER_ID) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.owner_id = owner_id

    def add(self, **kwargs: object) -> None:
        self.calls.append(("add", kwargs))

    def get_all(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_all", kwargs))
        return {"results": [{"id": "memory-1", "memory": "Likes tea"}]}

    def search(self, query: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "memory-1", "memory": "Likes tea", "score": 0.9}]}

    def get(self, memory_id: str) -> dict[str, str] | None:
        self.calls.append(("get", memory_id))
        return {"id": memory_id, "user_id": self.owner_id}

    def delete(self, memory_id: str) -> None:
        self.calls.append(("delete", memory_id))


class CommandTests(unittest.TestCase):
    def test_memories_and_search_use_current_user_filter(self) -> None:
        memory = FakeMemory()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            agent.handle_command(memory, "/memories", "", None)
            agent.handle_command(memory, "/search", "tea", None)

        self.assertIn("Likes tea", output.getvalue())
        self.assertEqual(memory.calls[0][0], "get_all")
        self.assertEqual(memory.calls[0][1]["filters"], {"user_id": agent.USER_ID})
        self.assertEqual(memory.calls[1][1], "tea")
        self.assertEqual(memory.calls[1][2]["filters"], {"user_id": agent.USER_ID})

    def test_forget_checks_owner_before_deleting(self) -> None:
        memory = FakeMemory()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            agent.handle_command(memory, "/forget", "memory-1", None)

        self.assertIn(("delete", "memory-1"), memory.calls)

    def test_forget_does_not_delete_another_users_memory(self) -> None:
        memory = FakeMemory(owner_id="another-user")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            agent.handle_command(memory, "/forget", "memory-1", None)

        self.assertIn("belongs to the current user", output.getvalue())
        self.assertNotIn(("delete", "memory-1"), memory.calls)

    def test_feedback_attaches_to_previous_exchange_and_is_persisted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "feedback.sqlite3"
            agent.record_feedback(
                "question",
                "answer",
                "bad",
                database,
                explanation="The answer was incomplete.",
                correction="Include the missing step.",
                verified=True,
            )

            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    """
                    SELECT user_id, rating, question, answer, explanation,
                           correction, verified
                    FROM feedback
                    """
                ).fetchone()

            self.assertEqual(
                row,
                (
                    agent.USER_ID,
                    "bad",
                    "question",
                    "answer",
                    "The answer was incomplete.",
                    "Include the missing step.",
                    1,
                ),
            )

        memory = FakeMemory()
        output = io.StringIO()
        with (
            patch(
                "builtins.input",
                side_effect=["wrong source", "Use the official docs", "yes"],
            ),
            patch("agent.record_feedback", return_value=12) as save_feedback,
            contextlib.redirect_stdout(output),
        ):
            agent.handle_command(memory, "/feedback", "bad", ("question 2", "answer 2"))

        save_feedback.assert_called_once_with(
            "question 2",
            "answer 2",
            "bad",
            explanation="wrong source",
            correction="Use the official docs",
            verified=True,
        )
        self.assertIn("Feedback saved (ID 12)", output.getvalue())

    def test_feedback_without_answer_is_not_saved(self) -> None:
        output = io.StringIO()
        with patch("agent.record_feedback") as save_feedback:
            with contextlib.redirect_stdout(output):
                agent.handle_command(FakeMemory(), "/feedback", "good", None)

        self.assertIn("There is no previous answer", output.getvalue())
        save_feedback.assert_not_called()

    def test_feedback_lessons_use_only_relevant_user_confirmed_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "feedback.sqlite3"
            agent.record_feedback(
                "How do I calculate portfolio return?",
                "Wrong calculation.",
                "bad",
                database,
                explanation="The answer used a simple rather than annualized return.",
                correction="Use the annualized return formula.",
                verified=True,
            )
            agent.record_feedback(
                "How do I calculate portfolio return?",
                "Correct calculation.",
                "good",
                database,
            )
            agent.record_feedback(
                "How do I bake sourdough bread?",
                "Wrong recipe.",
                "bad",
                database,
                correction="Use a longer fermentation.",
                verified=True,
            )

            context = agent.get_feedback_context(
                "Explain annualized portfolio return.", database
            )

        self.assertIn("annualized return formula", context)
        self.assertIn("Prior question: How do I calculate portfolio return?", context)
        self.assertNotIn("Correct calculation", context)
        self.assertNotIn("longer fermentation", context)

    def test_legacy_feedback_rows_migrate_as_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        rating TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO feedback
                        (user_id, rating, question, answer, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (agent.USER_ID, "good", "portfolio return", "answer", "now"),
                )

            context = agent.get_feedback_context("portfolio return", database)

            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    "SELECT explanation, correction, verified FROM feedback"
                ).fetchone()

        self.assertEqual(row, (None, None, 0))
        self.assertEqual(context, "")


class MemoryPolicyTests(unittest.TestCase):
    def test_respond_sends_only_user_input_with_durable_memory_policy(self) -> None:
        question = "I prefer concise answers and am learning Python."
        answer = "I'll keep answers concise and help with Python."
        memory = FakeMemory()
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content=answer))
        ]

        with patch("agent.get_memory_context", return_value=""), patch(
            "agent.get_feedback_context", return_value="- Use the annualized formula."
        ):
            result = agent.respond(client, memory, question)

        self.assertEqual(result, answer)
        self.assertEqual(
            memory.calls,
            [
                (
                    "add",
                    {
                        "messages": [{"role": "user", "content": question}],
                        "user_id": agent.USER_ID,
                        "prompt": agent.MEMORY_POLICY,
                    },
                )
            ],
        )
        self.assertIn("source or provenance", agent.MEMORY_POLICY)
        self.assertIn("exact reporting period", agent.MEMORY_POLICY)
        self.assertIn("do not store it", agent.MEMORY_POLICY)
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertIn("Use the annualized formula", system_prompt)
        self.assertIn("positive rating alone is not a verified fact", system_prompt)


if __name__ == "__main__":
    unittest.main()
