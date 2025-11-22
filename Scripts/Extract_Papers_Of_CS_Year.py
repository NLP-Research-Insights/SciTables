import json
from datetime import datetime

input_file = "../arxiv-metadata-oai-snapshot.json"
output_file = "cs_papers__2017.json"

cs_categories = {
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG",
    "cs.CL", "cs.CR", "cs.CV", "cs.CY", "cs.DB",
    "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET",
    "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC",
    "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA",
    "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI",
    "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO",
    "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
}

count =  0
with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    f_out.write("[\n")  # Start JSON array
    first_entry = True

    for line in f_in:
        try:
            paper = json.loads(line.strip())

            # Get categories as a space-separated string or a list
            categories = paper.get("categories", "")
            if isinstance(categories, str):
                category_list = categories.strip().split()
            else:
                category_list = categories  # assume it's already a list

            # Filter for CS categories
            if not any(cat in cs_categories for cat in category_list):
                continue

            # Extract year from first version's "created" field
            versions = paper.get("versions", [])
            if not versions:
                continue

            created_str = versions[0].get("created", "")
            if not created_str:
                continue

            # Parse date to extract year
            try:
                created_date = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z")
                year = created_date.year
            except ValueError:
                continue

            # Filter papers after 2000
            if year == 2017:
                if not first_entry:
                    f_out.write(",\n")
                json.dump(paper, f_out, indent=4)
                first_entry = False
                count += 1

        except json.JSONDecodeError:
            print("Skipping malformed line.")

    f_out.write("\n]\n")  # Close JSON array

print(f"CS papers in 2017 saved to {output_file}")
print(f"Total number of CS papers in 2017: {count}")
