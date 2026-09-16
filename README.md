# Neural Classification of Query-Product Relevance

Coursework implementation of ESCI four-class query-product relevance classification: **E**xact, **S**ubstitute, **C**omplement, and **I**rrelevant. It is a pairwise classification exercise, not a personalised recommender or full search-ranking system.

## Reproducible setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10--3.12 is recommended because the machine's default Python 3.15 alpha may not yet be supported by PyTorch. Obtain raw files following [data/README.md](data/README.md), then run:

```powershell
python run_pipeline.py --profile smoke
python run_pipeline.py --profile assignment
python predict.py --query "wireless mouse" --title "Bluetooth wireless mouse, black"
```

`smoke` validates the complete path on a small query-group sample and is never a formal result. `assignment` targets 40,000 development and 12,000 official-test-source rows; query-group sampling can make actual totals differ. If data are smaller, actual available rows are used without duplication.

## Method and safeguards

The pipeline filters `large_version == 1` and `product_locale == "us"` before joining by `(product_id, product_locale)`. It uses a left, many-to-one validated merge with an indicator; audits and cleaning logs are saved. Official test remains test-only. Within official train, normalized query groups form an 80/20 train/validation split. Any normalized query found in official test is removed from development to make the reported test an unseen-query evaluation.

The model uses shared training-only TF-IDF (unigrams/bigrams), training-only TruncatedSVD, pairwise vector operations and four small numeric features. It compares most-frequent Dummy, Logistic Regression, and a PyTorch MLP. Metrics include Accuracy, Macro-F1, Weighted-F1, class metrics and fixed E/S/C/I confusion matrices.

Generated outputs are under `outputs/`; the English report reads these actual artifacts and states TODO if no run exists.
