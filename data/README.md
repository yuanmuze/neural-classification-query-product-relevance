# Data acquisition

This project uses the Amazon Shopping Queries Dataset (ESCI), Task 2. Raw files are deliberately not committed and must remain unchanged after acquisition.

Source: https://github.com/amazon-science/esci-data (README links to https://github.com/amazon-research/esci-code/tree/main/shopping_queries_dataset). Accessed: 2026-09-16. The upstream README describes `large_version == 1` as the Task 2/3 version.

Expected paths:

```
data/raw/shopping_queries_dataset_examples.parquet
data/raw/shopping_queries_dataset_products.parquet
```

Windows PowerShell (Git LFS is required if Git reports LFS pointers):

```powershell
git lfs install
git clone https://github.com/amazon-research/esci-code.git .\data\upstream-esci-code
git -C .\data\upstream-esci-code lfs pull
Copy-Item .\data\upstream-esci-code\shopping_queries_dataset\shopping_queries_dataset_examples.parquet .\data\raw\
Copy-Item .\data\upstream-esci-code\shopping_queries_dataset\shopping_queries_dataset_products.parquet .\data\raw\
```

Alternatively, from the project root run `powershell -ExecutionPolicy Bypass -File .\scripts\acquire_data.ps1`. It refuses to overwrite an existing raw file.

Then run `python run_pipeline.py --profile smoke` or `--profile assignment`. The pipeline rejects Git LFS pointer files and does not substitute synthetic data.
