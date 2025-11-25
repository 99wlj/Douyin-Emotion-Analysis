import csv
import os
import re


def clean_csv(input_path: str, output_path: str) -> dict:
    """
    Clean a CSV file by:
    - Parsing with Python's csv reader (handles quoted multi-line fields)
    - Dropping any data rows whose column count != header column count
    - Stripping leading/trailing whitespace from all fields

    Returns a summary dict with counts.
    """
    summary = {
        "input": input_path,
        "output": output_path,
        "kept_rows": 0,
        "dropped_rows": 0,
        "expected_cols": None,
        "total_rows": 0,
    }

    if not os.path.exists(input_path):
        print(f"[WARN] Missing {input_path}, skipping.")
        return summary

    # Read with utf-8-sig to gracefully handle BOM if present.
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)

    if not rows:
        print(f"[WARN] Empty file: {input_path}")
        return summary

    header = rows[0]
    expected_cols = len(header)
    summary["expected_cols"] = expected_cols
    summary["total_rows"] = max(0, len(rows) - 1)

    cleaned_rows = [header]

    for idx, row in enumerate(rows[1:], start=2):  # 1-based line number including header
        # Keep row only if column count matches header
        if len(row) != expected_cols:
            summary["dropped_rows"] += 1
            continue
        # Normalize whitespace: remove internal newlines/tabs and collapse spaces
        normalized = [re.sub(r"\s+", " ", field).strip() for field in row]
        cleaned_rows.append(normalized)
        summary["kept_rows"] += 1

    # Write out normalized CSV with one record per line
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(cleaned_rows)

    print(
        f"Cleaned '{input_path}' → '{output_path}': kept {summary['kept_rows']} rows, "
        f"dropped {summary['dropped_rows']}, expected cols = {expected_cols}"
    )
    return summary


def main():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    files = [
        (os.path.join(raw_dir, "StructBERT_带情绪标签的文本数据.csv"), os.path.join(processed_dir, "StructBERTDataSet.csv")),
        (os.path.join(raw_dir, "多语言模型_带情绪标签的文本数据.csv"), os.path.join(processed_dir, "MultiLangDataSet.csv")),
        (os.path.join(raw_dir, "带情绪标签及互动量的文本数据.csv"), os.path.join(processed_dir, "InteractionDataSet.csv")),
    ]

    for src, dst in files:
        clean_csv(src, dst)

    # Optional: write a brief report file
    # Removed report file generation per request.


if __name__ == "__main__":
    main()