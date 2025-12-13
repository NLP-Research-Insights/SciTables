# SciTables

Enhance Table-to-Text by LLMs using scientific tables.

SciTables is a large-scale scientific dataset designed to support research on table-to-text generation, scientific reasoning, and factuality evaluation using large language models (LLMs). The dataset is constructed from scientific literature and pairs structured tables with contextual textual descriptions and metadata, comprising approximately **120K table–text pairs** spanning **10+ computer science domains**.

---

## Repository Structure

The repository is organized as follows:
```
SciTables/
├── Data/ # Dataset files (tables and description pairs by year)
│ └── (tracked with Git LFS due to large file sizes)
├── Scripts/ # Data processing, cleaning, generation, and LLM-as-judge scripts
├── .gitattributes # Git LFS configuration for large JSON files
├── .gitignore
├── LICENSE # MIT License (applies to code and dataset)
├── README.md
└── requirements.txt # Python dependencies
```

## Data Storage and Git LFS

Due to the large size of dataset files (e.g., JSON tables and annotations), this repository uses **Git Large File Storage (Git LFS)**.

Before cloning the repository, please ensure Git LFS is installed:

```bash
git lfs install
git clone https://github.com/NLP-Research-Insights/SciTables.git
```
Without Git LFS, dataset files may appear as pointer files instead of the actual data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Dataset License

The SciTables dataset is released under the MIT License.

If you use this dataset in academic work, we kindly request that you cite the SciTables paper or repository.

