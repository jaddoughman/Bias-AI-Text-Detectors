"""Unified detector wrapper for the MGT detection systems analysed in the study.

Exposes a single classify(text, model_name) -> (label, score) interface so the
loader scripts in this repo can iterate over detectors without per-detector
plumbing.

Detectors:
  - 'www.originality.ai'                              : Originality.AI free API
  - 'gptzero'                                         : GPTZero v2 API (requires GPTZERO_API_KEY)
  - 'www.zerogpt.com'                                 : ZeroGPT public API
  - 'www.quillbot.com'                                : Quillbot AI detector (Selenium, headed)
  - 'Hello-SimpleAI/chatgpt-detector-roberta'         : HuggingFace local
  - 'fakespot-ai/roberta-base-ai-text-detection-v1'   : HuggingFace local
  - 'openai-community/roberta-large-openai-detector'  : HuggingFace local
  - 'TrustSafeAI/RADAR-Vicuna-7B'                     : HuggingFace local
  - 'MayZhou/e5-small-lora-ai-generated-detector'     : HuggingFace local (excluded, >60% FPR)

HuggingFace detectors are loaded at the pinned revisions listed in HF_DETECTORS, and P(AI)
is read from the softmax index that each publisher designates as machine-generated
(S5 Table). Inputs are truncated once at 512 tokens.

Correction (R5): an earlier version of this file read the Hello-SimpleAI score under the
label "LABEL_1"; the checkpoint's config.json names its labels "Human"/"ChatGPT", so that
lookup returned 0 for every text. The study's own runs used a different code path that
read the "ChatGPT" label correctly (see README, "Corrections in this version").

Returned label is normalised to {"Human", "ChatGPT"}; score is P(AI) in [0, 1]
for HF models and Originality, percentage in [0, 100] for ZeroGPT/Quillbot,
matching the source APIs.
"""

import json
import os
import time

import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


GPTZERO_API_KEY = os.environ.get("GPTZERO_API_KEY", "")

# checkpoint -> (pinned revision, softmax index of "AI-generated", use slow tokenizer)
HF_DETECTORS = {
    "Hello-SimpleAI/chatgpt-detector-roberta":
        ("d2b342c61775d5dd0221808a79983ed3b86ffd86", 1, False),   # config id2label {0: Human, 1: ChatGPT}
    "fakespot-ai/roberta-base-ai-text-detection-v1":
        ("f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db", 1, True),    # config id2label {0: Human, 1: AI}
    "openai-community/roberta-large-openai-detector":
        ("38f3e0ccf205e9c4e0b38ae9d75ec948141bf832", 0, False),   # detector/server.py: fake, real = probs
    "TrustSafeAI/RADAR-Vicuna-7B":
        ("4ff1f23a69a36aa1df47b0933be6279f1b896c9b", 0, False),   # HF Space app.py: softmax[:, 0] is AI
    "MayZhou/e5-small-lora-ai-generated-detector":
        ("483fc4969592dc20e00e5130e7187b5dd25dbcc7", 1, False),   # model card: Label_1 = AI-generated
}


class TextDetector:
    def __init__(self):
        self._hf_cache = {}

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
            url = "https://api.zerogpt.com/api/detect/detectText"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.zerogpt.com",
                "Referer": "https://www.zerogpt.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/139.0.0.0 Safari/537.36"
                ),
            }
            response = requests.post(
                url, data=json.dumps({"input_text": text}), headers=headers, timeout=60
            )
            response.raise_for_status()
            fake_pct = response.json().get("data", {}).get("fakePercentage", 0)
            return ("ChatGPT" if fake_pct > 50 else "Human"), fake_pct

        if model_name == "www.originality.ai":
            url = "https://api.originality.ai/api/v2-tools/free-tools/ai-scan"
            headers = {
                "accept": "application/json, text/plain, */*",
                "accept-language": "en-US,en",
                "content-type": "application/json",
            }
            response = requests.post(
                url, headers=headers, data=json.dumps({"content": text}), timeout=60
            )
            response.raise_for_status()
            ai_score = float(response.json()["ai"])
            return ("ChatGPT" if ai_score > 0.5 else "Human"), ai_score

        if model_name == "www.quillbot.com":
            return self._classify_quillbot(text)

        if model_name in HF_DETECTORS:
            return self._classify_hf(text, model_name)

        raise ValueError(f"Unknown detector: {model_name}")

    def _load_hf(self, model_name: str):
        if model_name not in self._hf_cache:
            revision, _, slow = HF_DETECTORS[model_name]
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=not slow)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).eval()
            self._hf_cache[model_name] = (tokenizer, model)
        return self._hf_cache[model_name]

    def _classify_hf(self, text: str, model_name: str):
        """P(AI) from the publisher-designated softmax index; 512-token truncation."""
        tokenizer, model = self._load_hf(model_name)
        ai_index = HF_DETECTORS[model_name][1]
        encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            p_ai = torch.softmax(model(**encoded).logits.float(), dim=-1)[0, ai_index].item()
        return ("ChatGPT" if p_ai > 0.5 else "Human"), p_ai

    def _classify_quillbot(self, text: str):
        """Quillbot has no public API. We drive the web UI with Selenium."""
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By

        driver = webdriver.Chrome()
        try:
            driver.get("https://quillbot.com/ai-content-detector")
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()

            editor = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div[contenteditable='true'][id='aidr-input-editor']")
                )
            )
            editor.clear()
            editor.send_keys(text)

            detect_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[.//span[contains(text(), 'Detect AI')]]")
                )
            )
            time.sleep(1)
            detect_button.click()
            time.sleep(2)

            score_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.MuiBox-root.css-1xb5foi")
                )
            )
            ai_score = int(score_element.text)
        finally:
            driver.quit()

        return ("ChatGPT" if ai_score > 50 else "Human"), ai_score
