
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parent
QDRANT_PATH = ROOT / "data" / "financial_qdrant"

COLLECTION = "reliance_financial_reports"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

load_dotenv(ROOT / ".env")


def retrieve_passages(
    question: str,
    top_k: int = 5,
    financial_year: str | None = None,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")

    if not QDRANT_PATH.exists():
        raise RuntimeError(
            f"Financial Qdrant database not found: {QDRANT_PATH}. "
            "Run ingest.py first."
        )

    openai_client = OpenAI()
    qdrant_client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not qdrant_client.collection_exists(COLLECTION):
            raise RuntimeError(
                f"Collection '{COLLECTION}' does not exist. "
                "Run ingest.py first."
            )

        # Embed the user's question.
        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[question],
        )
        query_vector = embedding_response.data[0].embedding

        # Optionally restrict results to one financial year.
        query_filter = None

        if financial_year:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="financial_year",
                        match=models.MatchValue(
                            value=financial_year
                        ),
                    )
                ]
            )

        # Retrieve semantically similar report passages.
        response = qdrant_client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append({
                "score": point.score,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", ""),
                "financial_year": payload.get(
                    "financial_year", ""
                ),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "document_id": payload.get("document_id", ""),
            })

        return results

    finally:
        qdrant_client.close()


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Search Reliance annual report passages."
    )
    parser.add_argument(
        "question",
        help="Financial research question to search for",
    )
    parser.add_argument(
        "--year",
        dest="financial_year",
        help="Optional year, e.g. FY2023-24",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve (default: 5)",
    )

    args = parser.parse_args()

    try:
        results = retrieve_passages(
            question=args.question,
            top_k=args.top_k,
            financial_year=args.financial_year,
        )

        if not results:
            print("No matching passages found.")
            return

        for index, result in enumerate(results, start=1):
            print(f"\n{'=' * 65}")
            print(f"Result #{index}")
            print(f"Similarity score: {result['score']:.4f}")
            print(
                f"Source: {result['source_file']} | "
                f"{result['financial_year']} | "
                f"PDF page {result['page_number']}"
            )
            print(f"Chunk: {result['chunk_index']}")
            print(f"\n{result['text']}")

    except Exception as error:
        raise SystemExit(f"Retrieval failed: {error}") from error


if __name__ == "__main__":
    main()
