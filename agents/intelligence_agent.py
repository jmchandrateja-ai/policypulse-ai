from agents.base_agent import BaseAgent
from agents.state import AgentState
import os

DOMAIN_KEYWORDS = {
    "infrastructure": ["road", "pothole", "water", "electricity", "metro",
                       "bridge", "drainage", "footpath", "signal", "construction",
                       "pipe", "sewage", "streetlight", "footpath"],
    "healthcare":     ["hospital", "medicine", "doctor", "health", "treatment",
                       "ambulance", "insurance", "vaccine", "clinic", "patient",
                       "nurse", "surgery", "pharmacy"],
    "education":      ["school", "college", "fee", "nep", "student", "teacher",
                       "university", "exam", "scholarship", "syllabus", "books"],
    "taxation":       ["tax", "gst", "property tax", "income tax", "fine",
                       "penalty", "payment", "bill", "charge", "revenue"],
    "housing":        ["rent", "house", "flat", "pg", "accommodation", "eviction",
                       "landlord", "tenant", "building", "society", "slum"],
    "environment":    ["pollution", "waste", "garbage", "tree", "lake",
                       "water body", "air quality", "noise", "plastic", "dump"],
    "transportation": ["bus", "auto", "metro", "train", "traffic", "parking",
                       "bmtc", "ksrtc", "cab", "route", "signal", "commute"],
    "other":          []
}

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

OPPOSE_WORDS  = ["against", "oppose", "reject", "wrong", "unfair",
                 "unacceptable", "too high", "should not", "withdraw",
                 "cancel", "scrap", "burden", "ridiculous", "pathetic"]
SUPPORT_WORDS = ["support", "agree", "welcome", "appreciate", "needed",
                 "helpful", "good", "correct", "right", "excellent",
                 "commendable", "great initiative"]
URGENCY_HIGH   = ["urgent", "immediately", "emergency", "critical", "dying",
                  "cannot afford", "no water", "no electricity", "helpless",
                  "serious", "life threatening", "accident", "danger"]
URGENCY_MEDIUM = ["problem", "issue", "concern", "difficulty", "trouble",
                  "worried", "affected", "challenge", "delay", "pending",
                  "weeks", "months"]


class IntelligenceAgent(BaseAgent):

    def __init__(self):
        super().__init__("Intelligence Agent")
        self.model     = None
        self.tokenizer = None

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        local_path = "models/sentiment_model"
        hf_model   = "Chanduaiml12/policypulse-sentiment"
        if os.path.exists(local_path):
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model     = AutoModelForSequenceClassification.from_pretrained(hf_model)
        self.model.eval()

    def predict_sentiment(self, text: str) -> str:
        import torch
        import numpy as np
        if not self.model:
            self.load_model()
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        return SENTIMENT_LABELS[int(np.argmax(probs))]

    def classify_domain(self, text: str):
        text_lower = text.lower()
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        score = scores[best]
        if score == 0:
            return "other", 0.40
        return best, min(0.5 + score * 0.1, 0.99)

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting intelligence analysis")

        text = state.moderated_text

        # Domain classification
        state.domain, state.domain_confidence = self.classify_domain(text)

        # Routing decision
        if state.domain_confidence >= 0.75:
            state.routing_decision = "auto"
        elif state.domain_confidence >= 0.45:
            state.routing_decision = "review"
        else:
            state.routing_decision = "clarify"

        # Sentiment
        state.sentiment = self.predict_sentiment(text)

        # Stance
        text_lower = text.lower()
        opp = sum(1 for w in OPPOSE_WORDS  if w in text_lower)
        sup = sum(1 for w in SUPPORT_WORDS if w in text_lower)
        if opp > sup:   state.stance = "oppose"
        elif sup > opp: state.stance = "support"
        else:           state.stance = "neutral"

        # Urgency
        if any(w in text_lower for w in URGENCY_HIGH):
            state.urgency = "high"
        elif any(w in text_lower for w in URGENCY_MEDIUM):
            state.urgency = "medium"
        else:
            state.urgency = "low"

        self.log(state, "Intelligence complete",
                 f"Domain: {state.domain} | "
                 f"Sentiment: {state.sentiment} | "
                 f"Stance: {state.stance} | "
                 f"Urgency: {state.urgency}")

        return state
