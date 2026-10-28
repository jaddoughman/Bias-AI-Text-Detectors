# Bias in AI Text Detection Systems — code and per-text outputs

Code and data accompanying *"Algorithmic barriers to educational equity: Bias in AI text detection systems"* (Doughman & Khreich, PLOS ONE). This version of the repository corresponds to the **R5 revision** of the manuscript (November 2026). It contains

1. the detector wrapper and the scripts used to query detectors on each corpus,
2. the script used to re-run the locally run (open-source) detectors at pinned revisions,
3. the analysis scripts that produce every table and figure in the R5 manuscript and its supporting information, and
4. **per-text detector outputs** (text identifier, label and score for every detector) in `outputs/`, including the September 2025 outputs of the commercial detectors, which cannot be regenerated.

License: MIT (see `LICENSE`).

## Detectors

| Detector | Identifier | Access | Status in the analysis |
| --- | --- | --- | --- |
| Originality.AI | `www.originality.ai` | free API (queried Sept 2025) | analysed |
| GPTZero | `gptzero` | API key (queried Sept 2025) | analysed |
| ZeroGPT | `www.zerogpt.com` | API (queried Sept 2025) | analysed |
| Hello-SimpleAI | `Hello-SimpleAI/chatgpt-detector-roberta` | HF, rev `d2b342c6` | analysed |
| Fakespot | `fakespot-ai/roberta-base-ai-text-detection-v1` | HF, rev `f9cdb14d` | analysed |
| RoBERTa-large (OpenAI) | `openai-community/roberta-large-openai-detector` | HF, rev `38f3e0cc` | analysed (from R5) |
| RADAR | `TrustSafeAI/RADAR-Vicuna-7B` | HF, rev `4ff1f23a` | analysed (from R5) |
| e5-small-lora | `MayZhou/e5-small-lora-ai-generated-detector` | HF, rev `483fc496` | excluded (74.6% FPR > 60% rule) |
| Binoculars | — | — | identified, not evaluated |

Full revision hashes and the softmax index that each publisher designates as "AI-generated" are in `text_detector.py` (`HF_DETECTORS`) and in S5 Table. Three checkpoints (RoBERTa-large, RADAR, e5-small-lora) do not name their labels in `config.json`; the conventions come from the publishers' reference code (OpenAI `detector/server.py`: `fake, real = probs`; RADAR Space `app.py`: softmax index 0 = AI) or model card (e5: `Label_1` = AI).

## Corrections in this version (R5)

* **Hello-SimpleAI label lookup.** The previous `text_detector.py` read the Hello-SimpleAI score under `"LABEL_1"`, but the checkpoint's `config.json` names its labels `"Human"`/`"ChatGPT"`; that lookup returned 0 for every text. The study's runs did not use that code path: they used the pipeline label directly (the `else` branch of the original wrapper, reproduced in `score_local_detectors.py --mode original`). Re-running that wrapper reproduces 2,064 of the 2,078 published ICNALE labels and all 1,481 commonly scored Blog labels (S7 Table). The wrapper is fixed.
* **Inputs longer than 512 tokens.** The 2025 wrapper truncated at `max_position_embeddings` (514), decoded, and re-tokenised without truncation, so any text of more than 512 tokens raised an error and was skipped. This removed 2 ICNALE essays and 519 Blog posts from the open-source results. All locally run detectors are now scored on every text with a single 512-token truncation.
* **Excluded detectors.** `rerun_excluded_detectors.py` (August 2026) covered ICNALE only; `score_local_detectors.py` now runs all locally run detectors on all corpora.
* **Corpus sampling defaults.** `run_detectors_on_arxiv.py` previously defaulted to 200 abstracts per field with a 1,500-character minimum, and `run_detectors_on_blogs.py` to balancing by gender and age with a 200-character minimum. Neither matched the study. The study used 90 abstracts per field with no minimum length (GPTZero on all 990; ZeroGPT, Originality.AI and the 2025 Hello-SimpleAI/Fakespot runs on the first 45 per field), and 2,000 blog posts longer than 500 characters, 1,000 per gender (age not balanced). The defaults and docstrings now say so, and the exact samples are identified in `outputs/`.

## Layout

```
text_detector.py           wrapper for all detectors (commercial APIs + pinned HF checkpoints)
score_local_detectors.py   re-run HF detectors on a CSV of texts (modes: standard | original)
linguistic_features.py     the 12 lexical/syntactic/readability/structural features
run_detectors_on_*.py      corpus loaders used to query detectors (ICNALE, Blog, arXiv, DAIGT)
analysis/                  scripts that produce every R5 table and figure from outputs/
outputs/                   per-text outputs (see below)
supplementary/             earlier supplementary CSVs (S4a/S4b/S5, R4)
```

## Per-text outputs (`outputs/`)

| File | Rows | Key | Notes |
| --- | --- | --- | --- |
| `icnale_outputs.csv` | 2,080 | ICNALE file name | country, CEFR, prompt, 12 features, label/score per detector. ICNALE texts are not redistributed (ICNALE terms of use); obtain the corpus from https://language.sakura.ne.jp/icnale/ and join on file name. |
| `blog_outputs.csv` | 2,000 | post text | gender, age, label/score per detector |
| `arxiv_outputs.csv` | 988 | arXiv identifier | field, label/score per detector |
| `daigt_outputs.csv` | 1,000 | DAIGT v2 row id | 500 AI + 500 human essays |
| `dactyl_outputs.csv` | 880 | DACTYL id | 2024–2025 generators, locally run detectors only |
| `icnale_whitespace_outputs.csv` | 2,080 | ICNALE file name | line-break normalisation experiment |

For each detector: `<detector>_flag` (1 = classified AI-generated, 0 = human, empty = not scored) and `<detector>_score` (P(AI) for HF detectors; vendor score for commercial detectors). Commercial outputs are the September 2025 responses; texts the vendor did not score are empty, including GPTZero responses other than human/AI, which the collection script discarded (S8 Table).

## Reproducing the R5 analysis

```bash
pip install -r requirements.txt
# 1. (optional) re-score texts with the locally run detectors at the pinned revisions
#    (CPU is sufficient; about 3 texts/s for base models and 1 text/s for RoBERTa-large/RADAR)
python score_local_detectors.py --input texts.csv --text-col Text --id-col text_id \
    --detectors hello fakespot roberta_large radar e5 --mode standard --output scores.csv
#    --mode original re-implements the 2025 wrapper (decode/re-encode at 514 tokens) for verification
# 2. recompute the main tables and the conditional analyses from the per-text outputs
cd analysis && python reproduce_tables.py
```

`analysis/reproduce_tables.py` recomputes the country, region, CEFR, Indian B2+, Blog and arXiv FPR tables (with Wilson intervals) and the conditional analyses of Section 4.4.3 directly from `outputs/`. `analysis/full_pipeline/` contains the scripts used to produce every table, supporting table and figure of the R5 manuscript from the per-text files and the raw re-run outputs (they expect the working layout `data/`, `runs/`, `tables/` used during the revision; `build_master.py` documents how the September 2025 result files were joined by text).

## Data sources

* ICNALE Written Essays Plus v0.5 — https://language.sakura.ne.jp/icnale/ (registration required)
* Blog Authorship Corpus — https://huggingface.co/datasets/barilan/blog_authorship_corpus
* arXiv abstracts — https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021
* DAIGT v2 — https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
* DACTYL — https://huggingface.co/datasets/ShantanuT01/DACTYL (test split)

## Software versions used for the R5 re-runs

Python 3.13, torch 2.14 (CPU), transformers 4.46.3, tokenizers as installed with transformers 4.46.3, pandas 3.0, statsmodels 0.14, scikit-learn 1.x. The September 2025 runs were made from Jupyter notebooks (Python 3.9.6 according to the notebook metadata) with the library versions current at the time.
