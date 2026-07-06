"""
SciTable Data Cleaner
Drops records where:
  - referencing_paragraphs_cleaned is null, [], or a list of empty strings
  - table_content is null or empty string

Usage:
    python clean_scitable.py --input combined_table_2021.json
    python clean_scitable.py --input combined_table_2021.json --output combined_table_2021_cleaned.json
    python clean_scitable.py --input data.jsonl --output data_cleaned.jsonl
"""

import json
import argparse
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_records(records: list[dict], path: Path) -> None:
    ext = path.suffix.lower()
    with open(path, "w", encoding="utf-8") as f:
        if ext == ".jsonl":
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, indent=2, ensure_ascii=False)


def is_empty_cleaned(value) -> bool:
    """
    Returns True (i.e. should be dropped) if referencing_paragraphs_cleaned is:
      - None / missing
      - empty list []
      - list where every string is blank ("", "   ", etc.)
    """
    if value is None:
        return True
    if not isinstance(value, list) or len(value) == 0:
        return True
    if all(isinstance(s, str) and not s.strip() for s in value):
        return True
    return False


def is_empty_content(value) -> bool:
    """Returns True if table_content is None or blank string."""
    return value is None or (isinstance(value, str) and not value.strip())


# ── cleaner ───────────────────────────────────────────────────────────────────

def clean(records: list[dict]) -> tuple[list[dict], dict]:
    total = len(records)
    dropped_content  = 0
    dropped_cleaned  = 0
    kept = []

    for r in records:
        if is_empty_content(r.get("table_content")):
            dropped_content += 1
            continue
        if is_empty_cleaned(r.get("referencing_paragraphs_cleaned")):
            dropped_cleaned += 1
            continue
        kept.append(r)

    stats = {
        "total_input":              total,
        "dropped_missing_content":  dropped_content,
        "dropped_empty_cleaned":    dropped_cleaned,
        "total_dropped":            dropped_content + dropped_cleaned,
        "total_kept":               len(kept),
        "retention_rate":           f"{len(kept) / total * 100:.1f}%" if total else "0.0%",
    }
    return kept, stats


def print_report(stats: dict, input_path: Path, output_path: Path) -> None:
    w = 45
    print("─" * (w + 12))
    print(f"  {'Metric':<{w}} Value")
    print("─" * (w + 12))
    print(f"  {'Input file':<{w}} {input_path.name}")
    print(f"  {'Output file':<{w}} {output_path.name}")
    print("─" * (w + 12))
    print(f"  {'Total input records':<{w}} {stats['total_input']:,}")
    print(f"  {'Dropped — table_content empty/missing':<{w}} {stats['dropped_missing_content']:,}")
    print(f"  {'Dropped — ref_paragraphs_cleaned empty':<{w}} {stats['dropped_empty_cleaned']:,}")
    print(f"  {'Total dropped':<{w}} {stats['total_dropped']:,}")
    print(f"  {'Total kept':<{w}} {stats['total_kept']:,}")
    print(f"  {'Retention rate':<{w}} {stats['retention_rate']}")
    print("─" * (w + 12))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean SciTable JSON/JSONL dataset")
    parser.add_argument("--input",  required=True, help="Input file (.json or .jsonl)")
    parser.add_argument("--output", default=None,  help="Output file (default: <input>_cleaned.<ext>)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.stem + "_cleaned" + input_path.suffix
        )

    print(f"\nLoading {input_path} …")
    records = load_records(input_path)
    print(f"Loaded {len(records):,} records.\n")

    cleaned, stats = clean(records)

    print_report(stats, input_path, output_path)

    save_records(cleaned, output_path)
    print(f"\nSaved {stats['total_kept']:,} cleaned records → {output_path}\n")


if __name__ == "__main__":
    main()