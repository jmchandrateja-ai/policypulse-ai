from agents.base_agent import BaseAgent
from agents.state import AgentState


POSITIVE_WORDS = [
    "good", "great", "excellent", "helpful", "benefit", "improve",
    "support", "appreciate", "thank", "better", "happy", "glad"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "horrible", "unfair", "burden", "problem",
    "issue", "wrong", "fail", "poor", "corrupt", "useless", "waste",
    "high", "expensive", "unaffordable", "ruin", "destroy", "hurt"
]

OPPOSE_WORDS = [
    "against", "oppose", "reject", "disagree", "unfair", "wrong",
    "should not", "must not", "too high", "not acceptable", "unacceptable",
    "withdraw", "cancel", "scrap", "revoke"
]

SUPPORT_WORDS = [
    "support", "agree", "good decision", "correct", "right",
    "welcome", "appreciate", "needed", "necessary", "helpful"
]

URGENCY_HIGH = [
    "urgent", "immediately", "emergency", "critical", "dying",
    "cannot afford", "no water", "no electricity", "helpless", "serious"
]

URGENCY_MEDIUM = [
    "problem", "issue", "concern", "difficulty", "trouble",
    "worried", "affected", "impact", "challenge"
]


class SentimentAgent(BaseAgent):

    def __init__(self):
        super().__init__("Sentiment and Stance Agent")

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting sentiment and stance analysis")

        text = state.transcript.lower()

        # Sentiment
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)

        if pos > neg:
            state.sentiment = "positive"
        elif neg > pos:
            state.sentiment = "negative"
        else:
            state.sentiment = "neutral"

        # Stance
        opp = sum(1 for w in OPPOSE_WORDS if w in text)
        sup = sum(1 for w in SUPPORT_WORDS if w in text)

        if opp > sup:
            state.stance = "oppose"
        elif sup > opp:
            state.stance = "support"
        else:
            state.stance = "neutral"

        # Urgency
        if any(w in text for w in URGENCY_HIGH):
            state.urgency = "high"
        elif any(w in text for w in URGENCY_MEDIUM):
            state.urgency = "medium"
        else:
            state.urgency = "low"

        self.log(state, "Analysis complete",
                 f"Sentiment: {state.sentiment} | "
                 f"Stance: {state.stance} | "
                 f"Urgency: {state.urgency}")

        return state
