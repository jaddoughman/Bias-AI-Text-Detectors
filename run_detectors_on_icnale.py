"""Run MGT detectors on ICNALE WEP essays.

Expects a directory of one-essay-per-file .txt files with names following the
ICNALE convention:  W_<CCC>_<PROMPT>_<NNN>_<CEFR>.txt
  CCC    three-letter country code (UAE, JPN, IND, ...)
  PROMPT PTJ0 = part-time-job, SMK0 = smoking-ban
  NNN    essay number within the cell
  CEFR   A2_0, B1_1, B1_2, B2_0 (= B2+)

For each essay we run every detector listed in --detectors and write a single
CSV row with text + ICNALE metadata + linguistic features + per-detector
(label, score) columns.

Example:
  python run_detectors_on_icnale.py \\
      --input-dir  ICNALE_WEP_UAE/ \\
      --output     results/icnale_uae.csv \\
      --detectors  www.originality.ai Hello-SimpleAI/chatgpt-detector-roberta
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from text_detector import TextDetector
from linguistic_features import LinguisticFeatureExtractor


COUNTRY_CODES = {
    "BGD": "Bangladesh", "BRN": "Brunei", "CHN": "China", "ENS": "Native English",
    "HKG": "Hong Kong", "IDN": "Indonesia", "IND": "India", "JPN": "Japan",
    "KHM": "Cambodia", "KOR": "Korea", "LAO": "Laos", "MAS": "Malaysia",
    "MMR": "Myanmar", "NPL": "Nepal", "PAK": "Pakistan", "PHL": "Philippines",
    "SGP": "Singapore", "THA": "Thailand", "TWN": "Taiwan", "UAE": "UAE",
    "VNM": "Vietnam",
}

# Detectors used in the paper. Default omits GPTZero (paid API).
DEFAULT_DETECTORS = [
    "www.originality.ai",
    "www.zerogpt.com",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]


def parse_metadata(filename: str):
    """Extract (country, prompt, cefr) from an ICNALE essay filename."""
    stem = Path(filename).stem
    m = re.match(r"W_(\w{3})_(\w+?)_\d+_(\w+)$", stem)
    if not m:
        return {"Country": "Unknown", "Prompt": "Unknown", "Proficiency": "Unknown"}
    ccc, prompt, cefr = m.groups()
    return {
        "Country": COUNTRY_CODES.get(ccc, ccc),
        "Prompt": "part-time job" if prompt.startswith("PTJ") else (
            "smoking ban" if prompt.startswith("SMK") else prompt
        ),
        "Proficiency": cefr.replace("B2_0", "B2+").replace("A2_0", "A2"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Folder of ICNALE .txt essays")
    ap.add_argument("--output", required=True, help="Destination CSV path")
    ap.add_argument(
        "--detectors", nargs="+", default=DEFAULT_DETECTORS,
        help=f"Detector names to run. Default: {DEFAULT_DETECTORS}",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap number of essays (for smoke tests)")
    ap.add_argument("--api-delay", type=float, default=1.5,
                    help="Seconds to sleep between API calls (commercial detectors)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    essay_files = sorted(in_dir.glob("*.txt"))
    if args.limit:
        essay_files = essay_files[: args.limit]
    print(f"Found {len(essay_files)} essays in {in_dir}")

    detector = TextDetector()
    feat_extractor = LinguisticFeatureExtractor()

    rows = []
    for i, path in enumerate(essay_files, 1):
        text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not text:
            continue

        row = {"Filename": path.name, **parse_metadata(path.name), "Text": text,
               "Golden Label": "Human"}
        row.update(feat_extractor.extract_features(text).to_dict())

        for det in args.detectors:
            try:
                label, score = detector.classify(text, det)
                row[f"{det} Label"] = label
                row[f"{det} Score"] = score
            except Exception as e:
                row[f"{det} Label"] = "ERROR"
                row[f"{det} Score"] = None
                row[f"{det} Error"] = str(e)[:200]
            # rate-limit commercial APIs only
            if det.startswith("www.") or det == "gptzero":
                time.sleep(args.api_delay)

        rows.append(row)
        if i % 25 == 0 or i == len(essay_files):
            print(f"  [{i}/{len(essay_files)}] {path.name}")
            pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
