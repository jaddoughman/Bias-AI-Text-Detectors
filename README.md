# Bias in AI Text Detection Systems — Code

This repository accompanies the manuscript *"Bias in AI Text Detection Systems"* (PLOS ONE). It provides the code used to (1) query five MGT detection systems over four corpora, (2) extract the 12 linguistic features used in the linguistic-mechanisms analysis, and (3) compute the fairness metrics reported in the paper.

## Layout

```
github/
├── text_detector.py            # Unified wrapper around the six retained detectors
├── linguistic_features.py      # The 12 lexical/syntactic/readability/structural features
├── run_detectors_on_icnale.py  # ICNALE WEP pipeline (RQ1, RQ2)
├── run_detectors_on_blogs.py   # Blog Authorship Corpus pipeline (RQ3)
├── run_detectors_on_arxiv.py   # arXiv abstracts pipeline (RQ3)
├── run_detectors_on_daigt.py   # DAIGT v2 AI-generated pipeline (§4.5 TPR / asymmetry)
├── compute_bias_metrics.py     # FPR by group, bias range, Cohen's d, Separation Index
└── requirements.txt
```

The five detectors covered, with the model identifier expected by `--detectors`:

| Detector                                          | Identifier                                                   | Access            |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------- |
| Originality.AI                                    | `www.originality.ai`                                         | API, key required |
| GPTZero                                           | `gptzero`                                                    | API, key required |
| ZeroGPT                                           | `www.zerogpt.com`                                            | API, key required |
| Hello-SimpleAI/chatgpt-detector-roberta           | `Hello-SimpleAI/chatgpt-detector-roberta`                    | HuggingFace local |
| fakespot-ai/roberta-base-ai-text-detection-v1     | `fakespot-ai/roberta-base-ai-text-detection-v1`              | HuggingFace local |

## Install

```bash
pip install -r requirements.txt
```

The commercial detectors each require an API key exported in the environment:
`GPTZERO_API_KEY`, `ORIGINALITY_API_KEY`, `ZEROGPT_API_KEY`.

## Datasets

The pipelines expect publicly available distributions of each corpus:

- **ICNALE WEP** — request access at https://language.sakura.ne.jp/icnale/. The script consumes one-essay-per-file `.txt` files following the canonical naming convention `W_<CCC>_<PROMPT>_<NNN>_<CEFR>.txt`.
- **Blog Authorship Corpus** — https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus (`blogtext.csv`).
- **arXiv** — https://www.kaggle.com/datasets/Cornell-University/arxiv (metadata snapshot, CSV or JSONL).
- **DAIGT v2** — https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset (`train_v2_drcat_02.csv`).

## Reproducing the paper's results

### 1. False-positive audits on human-authored corpora (§4.1–§4.4)

ICNALE — produces RQ1 (cross-cultural) and RQ2 (proficiency) tables:

```bash
python run_detectors_on_icnale.py \
    --input-dir  /path/to/ICNALE_WEP/ \
    --output     results/icnale.csv
```

Blog Authorship Corpus — RQ3 native-English-informal baseline:

```bash
python run_detectors_on_blogs.py \
    --input  blogtext.csv \
    --output results/blogs.csv \
    --sample 2000 --balance-by gender age_group
```

arXiv abstracts — RQ3 academic-field bias:

```bash
python run_detectors_on_arxiv.py \
    --input  arxiv-metadata-oai-snapshot.json \
    --output results/arxiv.csv \
    --per-field 200 --min-chars 1500
```

### 2. True-positive validation on AI-generated text (§4.5)

```bash
python run_detectors_on_daigt.py \
    --input  train_v2_drcat_02.csv \
    --output results/daigt_ai.csv \
    --per-generator 100
```

### 3. Compute fairness metrics

The bias-range and Cohen's-d tables are derived from the per-essay CSVs:

```bash
# RQ1 — Cross-country FPR + linguistic effect size
python compute_bias_metrics.py --input results/icnale.csv \
    --group-col Country --feature "Avg Word Length"

# RQ2 — Proficiency
python compute_bias_metrics.py --input results/icnale.csv \
    --group-col Proficiency --feature "Avg Word Length"

# RQ3 — Academic field
python compute_bias_metrics.py --input results/arxiv.csv \
    --group-col Academic_Field --feature "Flesch Reading Ease"

# RQ3 — Blog gender / age
python compute_bias_metrics.py --input results/blogs.csv \
    --group-col Gender
python compute_bias_metrics.py --input results/blogs.csv \
    --group-col "Age Group"
```

## Output schema

Every pipeline writes a CSV with one row per text and the same general shape:

| Column                                | Source                                      |
| ------------------------------------- | ------------------------------------------- |
| Text                                  | Raw essay/post/abstract                     |
| Demographic / field columns           | Country, Proficiency, Gender, Age Group, Academic_Field, Generator … |
| Golden Label                          | Human or ChatGPT (DAIGT only)               |
| 12 linguistic feature columns         | `linguistic_features.py`                    |
| `<detector> Label`                    | Normalised to Human / ChatGPT               |
| `<detector> Score`                    | P(AI) in [0, 1] (HF & Originality) or % (ZeroGPT) |
| `<detector> Error`                    | Populated only on failure                   |

This schema is what `compute_bias_metrics.py` consumes.

## Reproducibility notes

- All samplers use `random_state=42`.
- HuggingFace detectors truncate input at the 512-token model maximum.
- All detectors are evaluated at the 0.5 classification threshold.
- Commercial detectors are rate-limited with `--api-delay` (default 1.5 s); checkpoints are written every 25 rows so a long run can be resumed.

## License

MIT.
