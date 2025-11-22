import os
import re
import json
from collections import defaultdict
from tqdm import tqdm 
import logging

base_folder = "./all_tex_files_2018"
output_filename = "sample_table_paragraphs_2018.json"



# Create logs folder
log_folder = "sample_table_paragraphs_logs"
output_folder = "sample_table_paragraphs_output"

os.makedirs(log_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Full paths
output_path = os.path.join(output_folder, output_filename)

# Set up logging to a file
output_filename = os.path.splitext(os.path.basename(output_path))[0]  # → "sample_table_paragraphs4"
log_filename = f"process_{output_filename}.log"
log_path = os.path.join(log_folder, log_filename)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    encoding="utf-8"
)

def read_file_with_fallback(path):
    encodings = ['utf-8', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode file: {path} using common encodings.")

def extract_caption(text, start_index):
    """
    Extract the content inside a LaTeX caption starting with '{' at start_index.
    Supports nested braces.
    Returns the inner caption text and the index after the closing brace.
    """
    if text[start_index] != '{':
        raise ValueError("Expected '{' at the start of caption")
    brace_level = 0
    i = start_index
    while i < len(text):
        if text[i] == '{':
            brace_level += 1
        elif text[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                return text[start_index + 1:i], i + 1
        i += 1

    raise ValueError("Unbalanced braces in caption block")

def extract_tables(latex_content):
    """
    Extract tables with captions and labels from LaTeX content,
    supporting nested braces in captions and any order of caption/label.
    """
    tables = {}
    pattern = re.compile(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', flags=re.DOTALL)
    table_blocks = pattern.findall(latex_content)

    for block in table_blocks:
        full_table = f"\\begin{{table}}{block}\\end{{table}}"

        # Find label
        label_match = re.search(r'\\label\{(tab:[^}]+)\}', block)
        if not label_match:
            continue
        label = label_match.group(1)

        # Find \caption
        caption_index = block.find(r'\caption')
        if caption_index == -1:
            continue

        brace_index = block.find('{', caption_index)
        if brace_index == -1:
            continue

        try:
            caption_text, _ = extract_caption(block, brace_index)
            if label not in tables:
                tables[label] = {
                    'content': full_table,
                    'caption': caption_text.strip()
                }
        except ValueError:
            continue

    return tables

def remove_inline_comments_only(text):
    """
    Remove inline % comments from LaTeX code, only outside of full-line comments.
    Keeps full-line % comments untouched and handles escaped % properly.
    """
    def remove_inline(line):
        result = []
        i = 0
        while i < len(line):
            if line[i] == '%':
                if i > 0 and line[i-1] == '\\':  # escaped %, keep going
                    result.append('%')
                    i += 1
                else:
                    break  # unescaped %, comment starts here
            else:
                result.append(line[i])
                i += 1
        return ''.join(result)

    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('%'):  # full-line comment, preserve
            new_lines.append(line)
        else:
            new_lines.append(remove_inline(line))
    return '\n'.join(new_lines)


def extract_paragraphs_referencing_tables(latex_content, table_labels):
    referencing_paragraphs = defaultdict(list)
    
    # Step 1: Remove unescaped % comments
    cleaned_content = re.sub(r'(?<!\\)%.*', '', latex_content)
    
    # Step 2: Split content into paragraphs
    paragraphs = re.split(r'\n\s*\n', cleaned_content)

    # Step 3: Define LaTeX reference commands and label prefixes
    ref_cmds = ['ref', 'autoref', 'cref']
    label_prefixes = ['tab']
    regex = r'\\(?:' + '|'.join(ref_cmds) + r')\{(' + '|'.join(label_prefixes) + r':[^\}]+)\}'

    # Step 4: Scan each paragraph
    for para in paragraphs:
        refs = re.findall(regex, para)
        for ref in refs:
            if ref in table_labels:
                stripped_para = para.strip()
                if stripped_para not in referencing_paragraphs[ref]:
                    referencing_paragraphs[ref].append(stripped_para)

    return referencing_paragraphs

def process_latex_files(folder):
    table_refs_map = defaultdict(lambda: {
        'filename': '',
        'caption': '',
        'content': '',
        'referencing_paragraphs': []
    })
    tex_files = [f for f in os.listdir(folder) if f.endswith(".tex")]
    #for filename in os.listdir(folder):
    for filename in tqdm(tex_files, desc="Processing LaTeX files", unit="file"):
        if filename.endswith(".tex"):
            
            file_path = os.path.join(folder, filename)
            latex_content = read_file_with_fallback(file_path)
            cleaned_content = remove_inline_comments_only(latex_content) # cleaned the contents
            tables = extract_tables(latex_content)
            references = extract_paragraphs_referencing_tables(cleaned_content, tables.keys())
            # get log information
            num_tables = len(tables)
            num_referenced = sum(1 for label in tables if references.get(label))
            logging.info(f"[{filename}] Tables found: {num_tables}, Referenced: {num_referenced}")

            for label, info in tables.items():
                table_refs_map[label]['filename'] = filename
                table_refs_map[label]['caption'] = info['caption']
                table_refs_map[label]['content'] = info['content']
                table_refs_map[label]['referencing_paragraphs'].extend(references.get(label, []))

                if references.get(label):
                    logging.info(f"  ↳ {label} was referenced in {len(references[label])} paragraph(s)")
                else:
                    logging.info(f"  ↳ {label} was not referenced")


    return table_refs_map

def main():
    table_refs_map = process_latex_files(base_folder)

    output_all = [
        {
            'filename': info['filename'],
            'table_label': label,
            'table_caption': info['caption'],
            'table_content': info['content'],
            'referencing_paragraphs': info['referencing_paragraphs']
        }
        for label, info in table_refs_map.items()
    ]

    referenced_once = sum(1 for v in table_refs_map.values() if len(v['referencing_paragraphs']) == 1)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_all, json_file, indent=2, ensure_ascii=False)

    print(f"Json file contains {len(output_all)} unique tables.")
    print(f"There are {referenced_once} tables that are referenced only once.")

if __name__ == "__main__":
    main()
