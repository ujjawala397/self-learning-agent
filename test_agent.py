import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent


class FakeMemory:
    def __init__(self, owner_id: str = agent.USER_ID) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.owner_id = owner_id

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
            agent.record_feedback("question", "answer", "good", database)

            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    "SELECT user_id, rating, question, answer FROM feedback"
                ).fetchone()

            self.assertEqual(row, (agent.USER_ID, "good", "question", "answer"))

        memory = FakeMemory()
        with patch("agent.record_feedback") as save_feedback:
            agent.handle_command(
                memory, "/feedback", "bad", ("question 2", "answer 2")
            )

        save_feedback.assert_called_once_with("question 2", "answer 2", "bad")

    def test_feedback_without_answer_is_not_saved(self) -> None:
        output = io.StringIO()
        with patch("agent.record_feedback") as save_feedback:
            with contextlib.redirect_stdout(output):
                agent.handle_command(FakeMemory(), "/feedback", "good", None)

        self.assertIn("There is no previous answer", output.getvalue())
        save_feedback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
