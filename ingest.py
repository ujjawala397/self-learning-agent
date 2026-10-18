
import os
import re
import uuid
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "data" / "reports" / "reliance"
QDRANT_PATH = ROOT / "data" / "financial_qdrant"

COLLECTION = "reliance_financial_reports"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 50

load_dotenv(ROOT / ".env")


def extract_financial_year(filename: str) -> str:
    """
    Supports filenames such as:
    RIL_FY2023-24.pdf
    RIL_FY2024-2025.pdf
    """
    match = re.search(
        r"(?:FY[\s_-]*)?(20\d{2})[\s_-]*"
        r"(20\d{2}|\d{2})",
        filename,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot determine financial year from {filename}. "
            "Rename it to RIL_FY2023-24.pdf, for example."
        )

    start_year = int(match.group(1))
    end_text = match.group(2)

    end_year = (
        int(end_text)
        if len(end_text) == 4
        else (start_year // 100) * 100 + int(end_text)
    )

    if end_year != start_year + 1:
        raise ValueError(
            f"Invalid financial year in filename: {filename}"
        )

    return f"FY{start_year}-{end_year % 100:02d}"


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_text(text: str) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Each chunk stays within its original PDF page.
    """
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()

        if chunk:
            chunks.append(chunk)

        if start + CHUNK_SIZE >= len(text):
            break

    return chunks


def extract_chunks(pdf_path: Path) -> list[dict]:
    financial_year = extract_financial_year(pdf_path.name)
    document_id = f"RIL:{financial_year}:{pdf_path.name}"

    chunks = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_text = normalize_text(
                page.get_text("text", sort=True)
            )

            if not page_text:
                print(
                    f"Warning: no extractable text on "
                    f"{pdf_path.name}, page {page_index + 1}"
                )
                continue

            for chunk_index, chunk_text in enumerate(
                split_text(page_text)
            ):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{document_id}:{page_index}:{chunk_index}",
                    )
                )

                chunks.append({
                    "id": point_id,
                    "text": chunk_text,
                    "metadata": {
                        "company": "Reliance Industries Limited",
                        "ticker": "RELIANCE",
                        "document_type": "annual_report",
                        "document_id": document_id,
                        "source_file": pdf_path.name,
                        "financial_year": financial_year,
                        "page_number": page_index + 1,
                        "page_index": page_index,
                        "chunk_index": chunk_index,
                    },
                })

    return chunks


def create_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def ingest_report(
    pdf_path: Path,
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
) -> None:
    print(f"\nProcessing: {pdf_path.name}", flush=True)

    chunks = extract_chunks(pdf_path)

    if not chunks:
        print("No text chunks extracted. Skipping.")
        return

    # Replace previously indexed chunks for this document.
    # This also removes stale chunks if the PDF has changed.
    document_id = chunks[0]["metadata"]["document_id"]

    qdrant_client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[item["text"] for item in batch],
        )

        points = []

        for item, embedding in zip(
            batch, response.data
        ):
            points.append(
                models.PointStruct(
                    id=item["id"],
                    vector=embedding.embedding,
                    payload={
                        "text": item["text"],
                        **item["metadata"],
                    },
                )
            )

        qdrant_client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

        print(
            f"Indexed {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks",
            flush=True,
        )

    print(
        f"Completed {pdf_path.name}: "
        f"{len(chunks)} chunks indexed.",
        flush=True,
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing.")

    if not REPORTS_DIR.exists():
        raise SystemExit(
            f"Reports folder not found: {REPORTS_DIR}"
        )

    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(
            f"No PDF files found in {REPORTS_DIR}"
        )

    print(f"Found {len(pdf_files)} PDF(s) in {REPORTS_DIR}", flush=True)
    for pdf_path in pdf_files:
        print(f"  - {pdf_path.name}", flush=True)

    print("Initializing OpenAI embeddings client...", flush=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    openai_client = OpenAI()
    print(f"Opening local Qdrant store at {QDRANT_PATH}...", flush=True)
    qdrant_client = QdrantClient(path=str(QDRANT_PATH))

    try:
        print(f"Preparing collection '{COLLECTION}'...", flush=True)
        create_collection(qdrant_client)

        for pdf_path in pdf_files:
            ingest_report(
                pdf_path,
                openai_client,
                qdrant_client,
            )

        count = qdrant_client.count(
            collection_name=COLLECTION,
            exact=True,
        ).count

        print(
            f"\nIngestion complete. "
            f"Total indexed chunks: {count}",
            flush=True,
        )
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    main()
