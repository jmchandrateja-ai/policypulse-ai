from agents.base_agent import BaseAgent
from agents.state import AgentState
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

STANCE_OPPOSE  = ["against", "oppose", "reject", "wrong", "unfair",
                  "unacceptable", "too high", "should not", "must not",
                  "withdraw", "cancel", "scrap", "burden", "ridiculous"]
STANCE_SUPPORT = ["support", "agree", "welcome", "appreciate", "needed",
                  "helpful", "good decision", "correct", "right", "excellent",
                  "great initiative", "commendable"]
URGENCY_HIGH   = ["urgent", "immediately", "emergency", "critical", "dying",
                  "cannot afford", "no water", "no electricity", "helpless",
                  "serious", "life threatening"]
URGENCY_MEDIUM = ["problem", "issue", "concern", "difficulty", "trouble",
                  "worried", "affected", "challenge", "delay", "pending"]

HF_MODEL = "Chanduaiml12/policypulse-sentiment"

class SentimentAgent(BaseAgent):

    def __init__(self):
        super().__init__("Sentiment and Stance Agent")
        self.model     = None
        self.tokenizer = None

    def load_model(self):
        self.log_print("Loading sentiment model from HuggingFace Hub...")
        local_path = "models/sentiment_model"
        if os.path.exists(local_path):
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            self.model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
        self.model.eval()

    def log_print(self, msg):
        print(f"  [{self.name}] {msg}")

    def predict_sentiment(self, text: str) -> str:
        if not self.model:
            self.load_model()
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        pred  = int(np.argmax(probs))
        return SENTIMENT_LABELS[pred]

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting sentiment and stance analysis")

        text = state.transcript if state.transcript else state.moderated_text
        text_lower = text.lower()

        # Sentiment
        state.sentiment = self.predict_sentiment(text)
        self.log(state, "Sentiment detected", state.sentiment)

        # Stance
        opp = sum(1 for w in STANCE_OPPOSE  if w in text_lower)
        sup = sum(1 for w in STANCE_SUPPORT if w in text_lower)
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

        self.log(state, "Analysis complete",
                 f"Sentiment: {state.sentiment} | "
                 f"Stance: {state.stance} | "
                 f"Urgency: {state.urgency}")

        return state
