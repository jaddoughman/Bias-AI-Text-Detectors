"""Unified detector wrapper for the five retained MGT detection systems.

Exposes a single classify(text, model_name) -> (label, score) interface so the
loader scripts in this repo can iterate over detectors without per-detector
plumbing.

Detectors:
  - 'www.originality.ai'                              : Originality.AI official API (requires ORIGINALITY_API_KEY)
  - 'gptzero'                                         : GPTZero v2 API (requires GPTZERO_API_KEY)
  - 'www.zerogpt.com'                                 : ZeroGPT official API (requires ZEROGPT_API_KEY)
  - 'Hello-SimpleAI/chatgpt-detector-roberta'         : HuggingFace local
  - 'fakespot-ai/roberta-base-ai-text-detection-v1'   : HuggingFace local

Returned label is normalised to {"Human", "ChatGPT"}; score is P(AI) in [0, 1]
for HF models and Originality, percentage in [0, 100] for ZeroGPT,
matching the source APIs.
"""

import json
import os

import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


GPTZERO_API_KEY = os.environ.get("GPTZERO_API_KEY", "")
ORIGINALITY_API_KEY = os.environ.get("ORIGINALITY_API_KEY", "")
ZEROGPT_API_KEY = os.environ.get("ZEROGPT_API_KEY", "")


class TextDetector:
    def classify(self, text: str, model_name: str):

        if model_name == "gptzero":
            if not GPTZERO_API_KEY:
                raise RuntimeError("GPTZERO_API_KEY environment variable is not set.")
            url = "https://api.gptzero.me/v2/predict/text"
            payload = {"document": text, "multilingual": False}
            headers = {
                "x-api-key": GPTZERO_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            doc = response.json()["documents"][0]
            ai_score = doc.get(
                "completely_generated_prob",
                doc.get("class_probabilities", {}).get("ai", 0),
            )
            label = (
                doc["document_classification"]
                .replace("HUMAN_ONLY", "Human")
                .replace("AI_ONLY", "ChatGPT")
            )
            return label, ai_score

        if model_name == "www.zerogpt.com":
            if not ZEROGPT_API_KEY:
                raise RuntimeError("ZEROGPT_API_KEY environment variable is not set.")
            url = "https://api.zerogpt.com/api/detect/detectText"
            headers = {
                "ApiKey": ZEROGPT_API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            response = requests.post(
                url, data=json.dumps({"input_text": text}), headers=headers, timeout=60
            )
            response.raise_for_status()
            fake_pct = response.json().get("data", {}).get("fakePercentage", 0)
            return ("ChatGPT" if fake_pct > 50 else "Human"), fake_pct

        if model_name == "www.originality.ai":
            if not ORIGINALITY_API_KEY:
                raise RuntimeError("ORIGINALITY_API_KEY environment variable is not set.")
            url = "https://api.originality.ai/api/v1/scan/ai"
            headers = {
                "X-OAI-API-KEY": ORIGINALITY_API_KEY,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            response = requests.post(
                url, headers=headers, data=json.dumps({"content": text}), timeout=60
            )
            response.raise_for_status()
            ai_score = float(response.json()["score"]["ai"])
            return ("ChatGPT" if ai_score > 0.5 else "Human"), ai_score

        if model_name == "fakespot-ai/roberta-base-ai-text-detection-v1":
            return self._classify_hf(text, model_name, ai_label="AI")

        if model_name == "Hello-SimpleAI/chatgpt-detector-roberta":
            return self._classify_hf(text, model_name, ai_label="LABEL_1")

        raise ValueError(f"Unknown detector: {model_name}")

    def _classify_hf(self, text: str, model_name: str, ai_label: str):
        """Generic HuggingFace text-classification inference."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            top_k=None,
        )

        max_length = getattr(model.config, "max_position_embeddings", 512)
        encoded = tokenizer.encode(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )
        truncated = tokenizer.decode(encoded[0], skip_special_tokens=True)

        scores_list = classifier(truncated)[0]
        scores = {x["label"]: x["score"] for x in scores_list}
        p_ai = float(scores.get(ai_label, 0.0))
        return ("ChatGPT" if p_ai > 0.5 else "Human"), p_ai
