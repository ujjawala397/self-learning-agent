import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf

import ingest


class IngestTests(unittest.TestCase):
    def test_extracts_chunks_from_fiscal_year_named_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "RIL_FY2024-25.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Reliance financial report text.")
            document.save(pdf_path)
            document.close()

            chunks = ingest.extract_chunks(pdf_path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["financial_year"], "FY2024-25")
        self.assertEqual(chunks[0]["metadata"]["page_number"], 1)
        self.assertIn("Reliance financial report text", chunks[0]["text"])

    def test_main_reports_discovery_and_initializes_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory) / "reports"
            reports_dir.mkdir()
            (reports_dir / "RIL_FY2025-26.pdf").touch()
            (reports_dir / "RIL_FY2024-25.pdf").touch()
            qdrant_path = Path(directory) / "qdrant"
            fake_qdrant = Mock()
            fake_qdrant.count.return_value = SimpleNamespace(count=0)
            output = io.StringIO()

            with (
                patch.object(ingest, "REPORTS_DIR", reports_dir),
                patch.object(ingest, "QDRANT_PATH", qdrant_path),
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                patch.object(ingest, "OpenAI", return_value=Mock()),
                patch.object(ingest, "QdrantClient", return_value=fake_qdrant),
                patch.object(ingest, "create_collection") as create_collection,
                patch.object(ingest, "ingest_report") as process_report,
                contextlib.redirect_stdout(output),
            ):
                ingest.main()

        text = output.getvalue()
        self.assertIn("Found 2 PDF(s)", text)
        self.assertIn("Initializing OpenAI embeddings client", text)
        self.assertIn("Opening local Qdrant store", text)
        self.assertLess(
            text.index("RIL_FY2024-25.pdf"),
            text.index("RIL_FY2025-26.pdf"),
        )
        self.assertEqual(process_report.call_count, 2)
        create_collection.assert_called_once_with(fake_qdrant)
        fake_qdrant.close.assert_called_once()

    def test_main_fails_clearly_when_input_folder_has_no_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            with (
                patch.object(ingest, "REPORTS_DIR", reports_dir),
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            ):
                with self.assertRaisesRegex(SystemExit, "No PDF files found"):
                    ingest.main()


if __name__ == "__main__":
    unittest.main()
