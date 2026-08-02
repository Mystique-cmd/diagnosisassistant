

import csv
import os
import re
from typing import Dict, List, Tuple


DEFAULT_SYMPTOMS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "symptoms.csv"
)

NEGATION_WORDS = {"no", "not", "denies", "denied", "without", "never", "n't"}


URGENCY_KEYWORDS = {
    "CRITICAL": [
        "can't breathe", "cannot breathe", "unconscious", "unresponsive",
        "severe chest pain", "collapsed", "seizure", "not breathing",
        "emergency", "chest pain and shortness of breath",
    ],
    "HIGH": [
        "severe", "worsening", "getting worse", "high fever",
        "can't keep anything down", "extreme", "unbearable",
        "shortness of breath",
    ],
    "MEDIUM": [
        "moderate", "persistent", "for several days", "for a few days",
    ],
    "LOW": [
        "mild", "slight", "a little", "minor", "just started",
    ],
}


class NLPProcessor:
    

    def __init__(self, symptoms_csv: str = DEFAULT_SYMPTOMS_CSV):
        self.symptoms_csv = symptoms_csv
        # canonical_name -> {'description':..., 'body_system':..., 'patterns': [...]}
        self.symptom_table: Dict[str, Dict] = {}
        # phrase -> canonical_name, sorted longest-phrase-first so
        # multi-word synonyms are matched before shorter substrings
        self._phrase_to_symptom: List[Tuple[str, str]] = []
        self._load_symptom_table()

    # Load the shared vocabulary from data/symptoms.csv
   
    def _load_symptom_table(self) -> None:
        if not os.path.exists(self.symptoms_csv):
            raise FileNotFoundError(
                f"Could not find symptoms.csv at '{self.symptoms_csv}'. "
                "This module depends on the shared vocabulary defined "
                "there — run build_data.py or check the path."
            )

        with open(self.symptoms_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["symptom_name"].strip()
                synonyms = [s.strip() for s in row["synonyms"].split(";") if s.strip()]
                # the canonical name itself (with underscores turned to
                # spaces) is always a valid match phrase too
                phrases = set(synonyms) | {name.replace("_", " ")}

                self.symptom_table[name] = {
                    "description": row["description"],
                    "body_system": row["body_system"],
                    "phrases": sorted(phrases, key=len, reverse=True),
                }
                for phrase in phrases:
                    self._phrase_to_symptom.append((phrase.lower(), name))

        # longest phrases first, globally, so "chest pain" is checked
        # before any single-word phrase that might be a substring of it
        self._phrase_to_symptom.sort(key=lambda pair: len(pair[0]), reverse=True)

    
    # Step: symptom extraction
   
    def extract_symptoms(self, text: str) -> Dict:
        """
        Scan free text for known symptom phrases (from symptoms.csv)
        and return the matched canonical symptom names, with basic
        negation handling (e.g. "no fever" does not count as fever).
        """
        text_lower = text.lower()
        matched: List[str] = []
        matched_phrases: Dict[str, str] = {}  # canonical_name -> phrase found
        negated: List[str] = []

        for phrase, symptom_name in self._phrase_to_symptom:
            if symptom_name in matched_phrases:
                continue  # already found this symptom via a longer phrase
            idx = text_lower.find(phrase)
            if idx == -1:
                continue

            if self._is_negated(text_lower, idx):
                negated.append(symptom_name)
                continue

            matched.append(symptom_name)
            matched_phrases[symptom_name] = phrase

        return {
            "symptoms": matched,
            "matched_phrases": matched_phrases,
            "negated_symptoms": negated,
        }

    def _is_negated(self, text_lower: str, phrase_start_idx: int, window: int = 20) -> bool:
        """Check for a negation word in the few words immediately
        before the matched phrase (a simple, local negation check —
        not full dependency parsing, but enough to catch "no fever",
        "denies any cough", "without chest pain", etc.)."""
        window_start = max(0, phrase_start_idx - window)
        preceding = text_lower[window_start:phrase_start_idx]
        preceding_words = re.findall(r"[a-z']+", preceding)
        return any(w in NEGATION_WORDS or w.endswith("n't") for w in preceding_words[-4:])

    
    # Step: urgency estimation
    
    def estimate_urgency(self, text: str) -> str:
        """Estimate an urgency tier (CRITICAL/HIGH/MEDIUM/LOW) from
        severity language in the text. Defaults to MEDIUM if nothing
        matches, since an unclear description shouldn't be silently
        treated as low-priority."""
        text_lower = text.lower()
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if any(keyword in text_lower for keyword in URGENCY_KEYWORDS[tier]):
                return tier
        return "MEDIUM"

    
    # Step: full processing pipeline
    
    def process(self, text: str) -> Dict:
        """Run the full text -> structured-facts pipeline."""
        extraction = self.extract_symptoms(text)
        urgency = self.estimate_urgency(text)

        return {
            "raw_text": text,
            "symptoms": extraction["symptoms"],
            "matched_phrases": extraction["matched_phrases"],
            "negated_symptoms": extraction["negated_symptoms"],
            "urgency": urgency,
            "symptom_count": len(extraction["symptoms"]),
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent. Reads free text off the
        percept (checks `raw_text` first, falls back to `text`) and
        returns a result dict other modules can build on: `symptoms`
        for ml_classifier.predict()/percept.symptoms, and `urgency`
        for planner.create_treatment_plan()."""
        text = getattr(percept, "raw_text", None) or getattr(percept, "text", "")
        result = self.process(text)
        result["summary"] = (
            f"Extracted {result['symptom_count']} symptom(s), "
            f"urgency={result['urgency']}"
        )
        return result



# Manual test / demo

if __name__ == "__main__":
    nlp = NLPProcessor()

    samples = [
        "I've had a really high fever for the last two days, and I can't "
        "smell anything at all. I also have a dry cough and I'm exhausted.",

        "My chest hurts and I can't breathe properly, it's getting worse "
        "and I'm sweating a lot.",

        "Just a mild headache and a bit of a sore, achy body, nothing serious.",

        "No fever, no cough, but I've been extremely thirsty and peeing a "
        "lot the past week, and my vision has been blurry.",
    ]

    for text in samples:
        print("-" * 70)
        print(f"Input: {text}")
        result = nlp.process(text)
        print(f"Symptoms found : {result['symptoms']}")
        print(f"Negated        : {result['negated_symptoms']}")
        print(f"Urgency        : {result['urgency']}")

    print("-" * 70)
    print("\nanalyze(percept) interface:")

    class FakePercept:
        def __init__(self, raw_text):
            self.raw_text = raw_text

    percept = FakePercept(
        "Severe chest pain and shortness of breath, started an hour ago."
    )
    analysis = nlp.analyze(percept)
    print(analysis["summary"])
    print(f"  symptoms -> {analysis['symptoms']}")
    print(f"  urgency  -> {analysis['urgency']}")