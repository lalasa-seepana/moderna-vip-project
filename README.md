# mRNA vaccine clinical safety and dosage intelligence

An implementation of the supplied PRD: patient-data validation and risk classification, self-attention, local guideline retrieval, structured LLM reports, and a three-panel results dashboard.

**Student engineering project output — not real clinical guidance.** The supplied safety reference is course material. The required `FDA_Safety_Warnings_Cited` field contains quotations from that reference and does not imply FDA authorship.

## Installation and execution

Python 3.14 with `venv` support is used for this environment. All installed dependency versions are pinned in `requirements.txt`. The main packages are NumPy, pandas, scikit-learn, CPU-only PyTorch, Matplotlib, Seaborn, Sentence Transformers, ChromaDB, LangChain and its text splitter, Pydantic, joblib, python-dotenv, pytest and notebook tools. No GPU, database server or cloud infrastructure is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure these variables in a private `.env` file:

```text
LLM_API_KEY=<your API key>
LLM_MODEL=<your model identifier>
LLM_BASE_URL=<your compatible API endpoint>
```

The `.env` file is excluded from version control and submission. Never put credentials in source code, notebooks or generated reports. The first retrieval run downloads the public `sentence-transformers/all-MiniLM-L6-v2` model; its revision is pinned in the source. Subsequent runs reuse the local cache. LLM calls require API access and may incur usage charges.

Run the project from its root:

```bash
source .venv/bin/activate
python -m src.pipeline
python -m pytest -q
```

The pipeline trains both classifiers, computes attention, builds or reloads the local semantic index, evaluates retrieval, generates five schema-validated LLM reports, and rebuilds the dashboard and portfolio report. It uses stable output names, replacing previous results on reruns rather than accumulating duplicates.

## Project files

- `data/patient_records.csv`: supplied 500-row dataset.
- `data/mrna_safety_guidelines.txt`: text extracted from the supplied reference document.
- `data/patient_records_clean.csv`: generated data with missing biomarker/history values imputed using training statistics.
- `notebooks/01_data_eda.ipynb`: data profiling, three visualizations and model comparison.
- `notebooks/02_attention_math.ipynb`: self-attention derivation, hand calculation, heatmap and interpretation.
- `src/`: the five modules and the helpers used to run them and build the required report.
- `tests/`: data, attention, retrieval, schema, safety and saved-artifact checks.
- `outputs/risk_model.joblib`: selected classifier and fitted preprocessing.
- `outputs/figures/model_metrics.json`: classification reports, confusion matrices and ROC coordinates for both models, plus the selected model name.
- `outputs/figures/`: both confusion-matrix images, the attention heatmap and the three-panel dashboard.
- `outputs/attention.npz`: computed attention matrix and sequence used by the dashboard.
- `outputs/chroma_store/`: persistent vector index and source-chunk metadata needed to reload and verify it.
- `outputs/retrieval_evaluation.md`: five query results with timings and an unsupported-query failure case.
- `outputs/reports/`: five validated JSON/Markdown report pairs.
- `outputs/latencies.json`: measured retrieval/report times used by the dashboard.
- `outputs/portfolio_report.md`: final project report with results, interpretation, limitations and reproduction commands.

The virtual environment and embedding cache are local runtime dependencies, not submission artifacts. They are ignored by version control. The runtime imports do not require the original Word documents after the provided text has been extracted.

## Individual modules and notebooks

```bash
python src/risk_model.py
python src/attention.py
python src/rag_index.py
python src/report_generator.py
python src/dashboard.py
python -m src.portfolio
```

Dashboard and portfolio commands consume saved artifacts. Module 4 alone reads the private LLM configuration and makes LLM API calls; Modules 1–3 and 5 run locally without that key. Open the notebooks in a notebook editor and choose `.venv/bin/python` as the kernel.

To query the persisted index:

```python
from src.rag_index import GuidelineIndex, query_guidelines
index = GuidelineIndex().load()
chunks, seconds = query_guidelines(index, "Can thawed product be refrozen?")
```

## Implementation choices

The literal history category `None` is preserved. Only blank cells are missing. The seed-42 stratified 80/20 split precedes imputation, scaling and encoding, preventing preprocessing leakage. Positive-class test F1 selects the model as specified; this selection use of the test set is discussed in the portfolio report.

The handout's four-base encoder acceptance example is supported by the low-level encoder, while application inputs must contain 5–50 valid bases. Attention uses fixed random weights and has no biological interpretation.

LangChain's token splitter produces guideline chunks containing 300–500 embedding tokens with 50-token overlap. Source offsets preserve exact quotations. Numeric dose values are converted from micrograms to milligrams. The 100-microgram review threshold is not treated as an absolute maximum. The report layer checks explicit maxima separately, rejects unsupported citations, verifies the exact Module 1 risk statement, and retries invalid responses exactly once.

Reports use conservative eligibility when the supplied evidence cannot establish a decision. A zero dose means no recommendation. Every output report contains the required disclaimer, and all figures and performance measures come from executed modules.
