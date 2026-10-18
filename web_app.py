import json
import os
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from openai import OpenAI

import agent


def create_app(client: Any = None, memory: Any = None) -> Flask:
    web_app = Flask(__name__)
    web_app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    services: dict[str, Any] = {"client": client, "memory": memory}

    def get_services() -> tuple[Any, Any]:
        if services["client"] is None:
            services["client"] = OpenAI()
        if services["memory"] is None:
            services["memory"] = agent.create_memory()
        return services["client"], services["memory"]

    @web_app.get("/")
    def index() -> str:
        return render_template("index.html")

    @web_app.post("/api/chat")
    def chat() -> Response | tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object with a question."), 400

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="Enter a question first."), 400
        question = question.strip()
        if len(question) > 10_000:
            return jsonify(error="Questions must be 10,000 characters or fewer."), 400

        def events():
            try:
                client, memory = get_services()
                for event in agent.stream_research_response(
                    client, memory, question
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:
                web_app.logger.exception("Streaming response failed")
                yield (
                    'data: {"error":"The response could not be completed. '
                    'Check the server logs and try again."}\n\n'
                )

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @web_app.post("/api/feedback")
    def feedback() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send feedback as a JSON object."), 400

        rating = payload.get("rating")
        question = payload.get("question")
        answer = payload.get("answer")
        if rating not in {"good", "bad"}:
            return jsonify(error="Rating must be 'good' or 'bad'."), 400
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="The question is required."), 400
        if not isinstance(answer, str) or not answer.strip():
            return jsonify(error="The answer is required."), 400
        if len(question) > 10_000 or len(answer) > 30_000:
            return jsonify(error="The question or answer is too long to save."), 400

        explanation = payload.get("explanation")
        correction = payload.get("correction")
        if explanation is not None and not isinstance(explanation, str):
            return jsonify(error="Explanation must be text."), 400
        if correction is not None and not isinstance(correction, str):
            return jsonify(error="Correction must be text."), 400
        explanation = explanation.strip()[:5_000] or None if explanation else None
        correction = correction.strip()[:10_000] or None if correction else None
        verified = payload.get("confirmed") is True and bool(explanation or correction)

        try:
            feedback_id = agent.record_feedback(
                question.strip(),
                answer.strip(),
                rating,
                explanation=explanation if rating == "bad" else None,
                correction=correction if rating == "bad" else None,
                verified=verified if rating == "bad" else False,
            )
        except ValueError as error:
            return jsonify(error=str(error)), 400

        return jsonify(feedback_id=feedback_id)

    return web_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8000")),
        debug=False,
    )
