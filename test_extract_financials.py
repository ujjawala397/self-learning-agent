import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import extract_financials


class FinancialTableExtractionTests(unittest.TestCase):
    def test_financial_year_parsing_supports_two_and_four_digit_end_years(self) -> None:
        self.assertEqual(
            extract_financials.get_financial_year("RIL_FY2023-24.pdf"),
            "FY2023-24",
        )
        self.assertEqual(
            extract_financials.get_financial_year("RIL-FY2024-2025.pdf"),
            "FY2024-25",
        )

    def test_invalid_financial_year_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot identify financial year"):
            extract_financials.get_financial_year("annual-report.pdf")
        with self.assertRaisesRegex(ValueError, "Invalid financial year"):
            extract_financials.get_financial_year("RIL_FY2023-25.pdf")

    def test_cell_cleanup_normalizes_whitespace_and_none(self) -> None:
        self.assertEqual(extract_financials.clean_cell("  Revenue\n  from ops "), "Revenue from ops")
        self.assertEqual(extract_financials.clean_cell(None), "")
        self.assertEqual(extract_financials.clean_cell(123), "123")

    def test_extract_tables_preserves_source_metadata_and_nonempty_rows(self) -> None:
        table = Mock()
        table.extract.return_value = [
            ["Particulars", "2024"],
            [" Revenue\nfrom operations ", "100"],
            [None, " "],
        ]
        page = Mock()
        page.find_tables.return_value = SimpleNamespace(tables=[table])
        document = [page]

        with (
            patch("extract_financials.pymupdf.open") as open_pdf,
            tempfile.TemporaryDirectory() as directory,
        ):
            open_pdf.return_value.__enter__.return_value = document
            pdf_path = Path(directory) / "RIL_FY2024-25.pdf"
            result = extract_financials.extract_tables(pdf_path)

        self.assertEqual(
            result,
            [
                {
                    "company": "Reliance Industries Limited",
                    "financial_year": "FY2024-25",
                    "source_file": "RIL_FY2024-25.pdf",
                    "page_number": 1,
                    "table_index": 0,
                    "rows": [
                        ["Particulars", "2024"],
                        ["Revenue from operations", "100"],
                    ],
                }
            ],
        )

    def test_main_writes_json_for_each_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "reports"
            output_dir = Path(directory) / "financial_data"
            input_dir.mkdir()
            (input_dir / "RIL_FY2024-25.pdf").touch()
            expected = [{"source_file": "RIL_FY2024-25.pdf", "rows": [["Value"]]}]
            output = io.StringIO()

            with (
                patch.object(extract_financials, "REPORTS_DIR", input_dir),
                patch.object(extract_financials, "OUTPUT_DIR", output_dir),
                patch.object(extract_financials, "extract_tables", return_value=expected),
                contextlib.redirect_stdout(output),
            ):
                extract_financials.main()

            saved = json.loads(
                (output_dir / "RIL_FY2024-25_tables.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved, expected)
        self.assertIn("Tables extracted: 1", output.getvalue())

    def test_main_reports_missing_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                extract_financials, "REPORTS_DIR", Path(directory) / "missing"
            ), patch.object(
                extract_financials, "FALLBACK_REPORTS_DIR", Path(directory) / "also-missing"
            ):
                with self.assertRaisesRegex(SystemExit, "Reports folder does not exist"):
                    extract_financials.main()


if __name__ == "__main__":
    unittest.main()
