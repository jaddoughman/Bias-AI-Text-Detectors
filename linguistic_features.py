"""Extract the 12 linguistic features used in the paper's RQ4 analysis.

Features (lexical, syntactic, readability, structural) match the definitions
in Methods §3.4. Reads a single text string and returns a pandas Series so the
loader scripts can apply this row-wise.
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


FEATURE_COLUMNS = [
    "Word Count",
    "Sentence Count",
    "Avg Word Length",
    "Lexical Diversity",
    "Avg Sentence Length",
    "Flesch Reading Ease",
    "Flesch-Kincaid Grade Level",
    "Gunning Fog Index",
    "Paragraph Count",
    "Avg Sentences per Paragraph",
    "Avg Words per Paragraph",
    "Avg Paragraph Length",
]


class LinguisticFeatureExtractor:
    def extract_features(self, text: str) -> pd.Series:
        if not isinstance(text, str) or not text.strip():
            return pd.Series({col: 0 for col in FEATURE_COLUMNS})

        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        word_count = len(words)
        sentence_count = len(sentences)

        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count else 0
        )
        lexical_diversity = len(set(words)) / word_count if word_count else 0
        avg_sentence_length = word_count / sentence_count if sentence_count else 0

        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)
        sentences_per_paragraph = [len(sent_tokenize(p)) for p in paragraphs]
        words_per_paragraph = [len(word_tokenize(p)) for p in paragraphs]
        paragraph_chars = [len(p) for p in paragraphs]

        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else 0

        return pd.Series(
            {
                "Word Count": word_count,
                "Sentence Count": sentence_count,
                "Avg Word Length": avg_word_length,
                "Lexical Diversity": lexical_diversity,
                "Avg Sentence Length": avg_sentence_length,
                "Flesch Reading Ease": flesch_reading_ease(text),
                "Flesch-Kincaid Grade Level": flesch_kincaid_grade(text),
                "Gunning Fog Index": gunning_fog(text),
                "Paragraph Count": paragraph_count,
                "Avg Sentences per Paragraph": _mean(sentences_per_paragraph),
                "Avg Words per Paragraph": _mean(words_per_paragraph),
                "Avg Paragraph Length": _mean(paragraph_chars),
            }
        )
