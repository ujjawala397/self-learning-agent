import json
import re
from pathlib import Path
from typing import Any

import pymupdf


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "data" / "reports" / "reliance"
FALLBACK_REPORTS_DIR = ROOT / "static" / "reliance"
OUTPUT_DIR = ROOT / "financial_data"


def get_financial_year(filename: str) -> str:
    match = re.search(
        r"(20\d{2})[\s_-]*(20\d{2}|\d{2})",
        filename,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot identify financial year from {filename}. "
            "Include the years in the filename, e.g. RIL_FY2023-24.pdf"
        )

    start_year = int(match.group(1))
    end_text = match.group(2)
    end_year = (
        int(end_text)
        if len(end_text) == 4
        else (start_year // 100) * 100 + int(end_text)
    )

    if end_year != start_year + 1:
        raise ValueError(f"Invalid financial year in filename: {filename}")

    return f"FY{start_year}-{end_year % 100:02d}"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def extract_tables(pdf_path: Path) -> list[dict[str, Any]]:
    financial_year = get_financial_year(pdf_path.name)
    extracted = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            try:
                tables = page.find_tables().tables
            except Exception as error:
                print(
                    f"Could not extract tables from {pdf_path.name}, "
                    f"page {page_index + 1}: {error}"
                )
                continue

            for table_index, table in enumerate(tables):
                cleaned_rows = [
                    [clean_cell(cell) for cell in row]
                    for row in table.extract()
                ]
                cleaned_rows = [
                    row for row in cleaned_rows if any(cell for cell in row)
                ]
                if not cleaned_rows:
                    continue

                extracted.append(
                    {
                        "company": "Reliance Industries Limited",
                        "financial_year": financial_year,
                        "source_file": pdf_path.name,
                        "page_number": page_index + 1,
                        "table_index": table_index,
                        "rows": cleaned_rows,
                    }
                )

    return extracted


def get_report_files() -> tuple[Path, list[Path]]:
    if REPORTS_DIR.exists():
        pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))
        if pdf_files:
            return REPORTS_DIR, pdf_files

    fallback_files = sorted(FALLBACK_REPORTS_DIR.glob("*.pdf"))
    if fallback_files:
        print(
            f"No PDFs found in {REPORTS_DIR}; using existing PDFs in "
            f"{FALLBACK_REPORTS_DIR}.",
            flush=True,
        )
        return FALLBACK_REPORTS_DIR, fallback_files

    if not REPORTS_DIR.exists():
        raise SystemExit(f"Reports folder does not exist: {REPORTS_DIR}")
    raise SystemExit(f"No PDF reports found in {REPORTS_DIR}")


def main() -> None:
    reports_dir, pdf_files = get_report_files()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading reports from: {reports_dir}", flush=True)
    for pdf_path in pdf_files:
        print(f"\nExtracting tables: {pdf_path.name}", flush=True)
        try:
            tables = extract_tables(pdf_path)
        except (OSError, ValueError) as error:
            print(f"Failed to extract {pdf_path.name}: {error}", flush=True)
            continue

        output_path = OUTPUT_DIR / f"{pdf_path.stem}_tables.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(tables, file, ensure_ascii=False, indent=2)

        print(f"Tables extracted: {len(tables)}", flush=True)
        print(f"Saved to: {output_path}", flush=True)
        if not tables:
            print(
                "Warning: No tables extracted. This PDF may require "
                "a different extraction method.",
                flush=True,
            )


if __name__ == "__main__":
    main()
