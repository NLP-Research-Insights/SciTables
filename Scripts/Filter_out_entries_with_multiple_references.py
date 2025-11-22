import json
import os

# Create output directory
output_dir = "json_files_after_filter"
os.makedirs(output_dir, exist_ok=True)

combined_filtered = []

for year in range(2017, 2024):
    input_filename = f"./sample_table_paragraphs_output/sample_table_paragraphs_{year}.json"
    output_filename = os.path.join(output_dir, f"referenced_once_tables_{year}.json")

    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter: only tables referenced once
        filtered = [entry for entry in data if len(entry["referencing_paragraphs"]) == 1]

        # Save per-year filtered JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

        combined_filtered.extend(filtered)
        print(f" Year {year}: {len(filtered)} entries saved to {output_filename}")

    except FileNotFoundError:
        print(f" Warning: {input_filename} not found. Skipping.")

# Save combined file
combined_output_path = os.path.join(output_dir, "referenced_once_tables_combined.json")
with open(combined_output_path, "w", encoding="utf-8") as f:
    json.dump(combined_filtered, f, indent=2, ensure_ascii=False)

print(f"\n All done! Combined {len(combined_filtered)} entries saved to {combined_output_path}")
