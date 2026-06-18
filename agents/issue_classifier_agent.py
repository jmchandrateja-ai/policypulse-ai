from agents.base_agent import BaseAgent
from agents.state import AgentState


DOMAIN_KEYWORDS = {
    "education": ["school", "college", "fee", "nep", "student", "teacher",
                  "university", "exam", "scholarship", "syllabus"],
    "healthcare": ["hospital", "medicine", "doctor", "health", "treatment",
                   "ambulance", "insurance", "vaccine", "clinic", "patient"],
    "infrastructure": ["road", "pothole", "water", "electricity", "metro",
                       "bridge", "drainage", "footpath", "signal", "construction"],
    "taxation": ["tax", "gst", "property tax", "income tax", "fine",
                 "penalty", "payment", "bill", "charge", "fee"],
    "housing": ["rent", "house", "flat", "pg", "accommodation", "eviction",
                "landlord", "tenant", "building", "society"],
    "environment": ["pollution", "waste", "garbage", "tree", "lake",
                    "water body", "air quality", "noise", "plastic", "dump"],
    "transportation": ["bus", "auto", "metro", "train", "traffic", "parking",
                       "bmtc", "ksrtc", "cab", "route"],
    "other": []
}


class IssueClassifierAgent(BaseAgent):

    def __init__(self):
        super().__init__("Issue Classifier Agent")

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting issue classification")

        text = state.transcript.lower()
        scores = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[domain] = score

        # Pick domain with highest score
        best_domain = max(scores, key=scores.get)
        best_score  = scores[best_domain]

        if best_score == 0:
            state.domain = "other"
            state.domain_confidence = 0.4
        else:
            state.domain = best_domain
            state.domain_confidence = min(0.5 + best_score * 0.1, 0.99)

        # Routing decision
        if state.domain_confidence >= 0.75:
            state.routing_decision = "auto"
        elif state.domain_confidence >= 0.45:
            state.routing_decision = "review"
        else:
            state.routing_decision = "clarify"

        self.log(state, "Classification complete",
                 f"Domain: {state.domain} | "
                 f"Confidence: {state.domain_confidence:.2f} | "
                 f"Routing: {state.routing_decision}")

        return state
