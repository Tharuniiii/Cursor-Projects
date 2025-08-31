from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from langdetect import detect, DetectorFactory
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


# Make langdetect deterministic
DetectorFactory.seed = 42


@dataclass
class SentenceScore:
    sentence: str
    true_score: float  # probability the sentence is true


def map_langdetect_to_m2m(lang_code: str) -> str:
    """Map langdetect language code to M2M100 language code.

    Falls back to two-letter code when unknown. Special-cases some common variants.
    """
    if not lang_code:
        return "en"
    lang_code = lang_code.lower()
    special_map = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh": "zh",
        "pt-br": "pt",
        "sr": "sr",
        "jw": "jv",  # langdetect sometimes returns jw for Javanese
        "he": "he",  # M2M uses he for Hebrew
        "iw": "he",  # legacy code for Hebrew
        "id": "id",
        "fil": "tl",
        "uk": "uk",
        "no": "no",
        "nb": "no",
        "nn": "no",
    }
    if lang_code in special_map:
        return special_map[lang_code]
    # Default to first two chars; M2M uses 2-letter ISO codes for most languages
    return lang_code[:2]


class FakeNewsService:
    """Backend service for multilingual fake news detection with translation and highlighting."""

    def __init__(self) -> None:
        # Translation model (multilingual to English)
        self.translation_model_name = "facebook/m2m100_418M"
        self.trans_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name)

        # Zero-shot classifier (general purpose)
        # Using BART MNLI for robust zero-shot classification
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )

        # Candidate labels for truth assessment
        self.candidate_labels = ["true", "false"]
        self.hypothesis_template = "This statement is {}."

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "en"

    def translate_to_english(self, text: str, src_lang: str) -> str:
        if not text.strip():
            return ""
        if src_lang.startswith("en"):
            return text

        m2m_src = map_langdetect_to_m2m(src_lang)
        # Configure tokenizer for source language
        self.trans_tokenizer.src_lang = m2m_src
        encoded = self.trans_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        generated_tokens = self.trans_model.generate(
            **encoded,
            forced_bos_token_id=self.trans_tokenizer.get_lang_id("en"),
            max_new_tokens=1024,
        )
        translated = self.trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated[0]

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        if not text.strip():
            return []
        # Handle common sentence terminators including CJK
        # First, ensure that CJK terminators are followed by a space to help splitting
        text = re.sub(r"([。！？])", r"\1 ", text)
        parts = re.split(r"(?<=[\.!?。！？])\s+", text.strip())
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    def classify_truth_score(self, text: str) -> float:
        if not text.strip():
            return 0.5
        result = self.zero_shot(
            text,
            self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )
        # result has fields: labels, scores; labels aligned with scores
        label_to_score: Dict[str, float] = {l.lower(): s for l, s in zip(result["labels"], result["scores"])}
        # Normalize in case order changes
        true_score = float(label_to_score.get("true", 0.0))
        return true_score

    def classify_sentences(self, sentences: List[str], max_sentences: int = 40) -> List[SentenceScore]:
        if not sentences:
            return []
        limited = sentences[:max_sentences]
        results = self.zero_shot(
            limited,
            self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )
        scores: List[SentenceScore] = []
        # results is a dict if input is str, list of dicts if input is list
        for sent, res in zip(limited, results):
            label_to_score = {l.lower(): s for l, s in zip(res["labels"], res["scores"])}
            scores.append(SentenceScore(sentence=sent, true_score=float(label_to_score.get("true", 0.0))))

        # For remaining sentences beyond limit, mark neutral
        for sent in sentences[len(limited):]:
            scores.append(SentenceScore(sentence=sent, true_score=0.5))
        return scores

    @staticmethod
    def verdict_from_score(true_score: float) -> Tuple[str, str]:
        """Return (label, color_hex) based on score."""
        if true_score >= 0.6:
            return "Likely True", "#16a34a"  # green-600
        if true_score <= 0.4:
            return "Likely False", "#dc2626"  # red-600
        return "Uncertain", "#d97706"  # amber-600

    @staticmethod
    def sentence_bg_color(true_score: float) -> str:
        # Map score to color: red (false) to green (true), yellow near 0.5
        # Simple interpolation between red and green via hue at 0.33*score
        # For readability, use discrete thresholds
        if true_score >= 0.7:
            return "#bbf7d0"  # green-200
        if true_score >= 0.6:
            return "#dcfce7"  # green-100
        if true_score <= 0.3:
            return "#fecaca"  # red-200
        if true_score <= 0.4:
            return "#fee2e2"  # red-100
        return "#fef9c3"  # yellow-100

    def build_highlighted_html(self, sentence_scores: List[SentenceScore]) -> str:
        spans: List[str] = []
        for s in sentence_scores:
            color = self.sentence_bg_color(s.true_score)
            conf = int(round(s.true_score * 100))
            span = (
                f'<span style="background:{color}; padding:2px 4px; border-radius:4px; margin-right:2px; display:inline;">'
                f"{self._escape_html(s.sentence)} "
                f"<small style='opacity:0.7'>(true {conf}%)</small>"
                "</span>"
            )
            spans.append(span)
        return "<div style='line-height:1.9'>" + " ".join(spans) + "</div>"

    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def process(self, article_text: str) -> Dict[str, object]:
        article_text = (article_text or "").strip()
        if not article_text:
            return {
                "detected_language": "",
                "translated_text": "",
                "overall_true_score": 0.5,
                "verdict_label": "",
                "verdict_color": "#6b7280",
                "highlighted_html": "",
                "sentence_scores": [],
            }

        lang = self.detect_language(article_text)
        translated = self.translate_to_english(article_text, lang)
        overall_score = self.classify_truth_score(translated)
        verdict_label, verdict_color = self.verdict_from_score(overall_score)

        sentences = self.split_into_sentences(translated)
        sent_scores = self.classify_sentences(sentences)
        highlighted_html = self.build_highlighted_html(sent_scores)

        return {
            "detected_language": lang,
            "translated_text": translated,
            "overall_true_score": overall_score,
            "verdict_label": verdict_label,
            "verdict_color": verdict_color,
            "highlighted_html": highlighted_html,
            "sentence_scores": [
                {"sentence": s.sentence, "true_score": s.true_score} for s in sent_scores
            ],
        }


