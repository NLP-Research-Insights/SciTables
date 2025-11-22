import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from huggingface_hub import login

def extract_converted_text(full_output: str) -> str:
    match = re.search(r"Answer\s*:\s*(.*?)(\[END\])", full_output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()  # returns text between 'Answer:' and '[END]'
    return ""

# === Config ===
model_id = "meta-llama/Llama-3.3-70B-Instruct"
hf_token = "XXX"
input_path = "nlp_project/sample_table_paragraphs_2018_oneparag_100x500.json"

print(f"Using model: {model_id}") # for debbug
print(f"Using token: {hf_token[:8]}...")  # safe to log beginning


output_dir = "./output_batches" 
os.makedirs(output_dir, exist_ok=True)

# === Login
login(token=hf_token)

# === Load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# === Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total input samples: {len(data)}")

batch_size = 100
total_batches = (len(data) + batch_size - 1) // batch_size

existing_files = set(os.listdir(output_dir))
for batch_num in range(total_batches):
    output_file = f"batch_{batch_num + 1}.json"
    if output_file in existing_files:
        print(f"[SKIP] {output_file} already exists. Skipping.")
        continue
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(data))
    batch_data = data[start_idx:end_idx]

    for entry in tqdm(batch_data, desc=f"Processing batch {batch_num + 1}/{total_batches}"):
        paragraphs = entry.get("referencing_paragraphs", [])
        cleaned_paragraphs = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                cleaned_paragraphs.append("")
                continue

            prompt = (
                f"<s>[INST] Make the following paragraph LaTeX command free and convert it to plain English text. "
                f"Keep everything else the same. Do not do paraphrasing. Add the [END] token at the end of the cleaned paragraph. "
                f"{p}\n\n[/INST] Answer:"
            )

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=200,  # generates 200 tokens after prompt,
                    #max_length=1000
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            cleaned = extract_converted_text(output_text)
            cleaned_paragraphs.append(cleaned)

        entry["referencing_paragraphs_cleaned"] = cleaned_paragraphs

    # === Save each batch separately
    output_file = os.path.join(output_dir, f"batch_{batch_num + 1}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print(f" Batch {batch_num + 1} saved to: {output_file}")
