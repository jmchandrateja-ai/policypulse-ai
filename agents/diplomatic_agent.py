from agents.base_agent import BaseAgent
from agents.state import AgentState
import re

# Words that suggest personal political attacks
POLITICAL_ATTACK_PATTERNS = [
    r'\b(modi|rahul|kejriwal|yogi|siddaramaiah|bommai|bjp|congress|aap)\b',
    r'\b(corrupt|criminal|thief|liar|murderer|traitor)\b',
    r'\bkill\b|\bhang\b|\barrest\b.*\bpolitician\b',
]

DOMAIN_EMOJIS = {
    "infrastructure": "🏗️",
    "healthcare":     "🏥",
    "education":      "📚",
    "taxation":       "💰",
    "housing":        "🏠",
    "environment":    "🌿",
    "transportation": "🚌",
    "other":          "📋",
}

URGENCY_COLORS = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
}


class DiplomaticAgent(BaseAgent):

    def __init__(self):
        super().__init__("Diplomatic Filter Agent")

    def remove_political_attacks(self, text: str) -> str:
        for pattern in POLITICAL_ATTACK_PATTERNS:
            text = re.sub(pattern, "[authority]", text,
                          flags=re.IGNORECASE)
        return text.strip()

    def generate_public_card(self, state: AgentState) -> str:
        emoji   = DOMAIN_EMOJIS.get(state.domain, "📋")
        urgency = URGENCY_COLORS.get(state.urgency, "🟢")
        nearby  = state.nearby_count

        card = f"""{emoji} {state.domain.upper()} ISSUE — {state.area}
{urgency} Urgency: {state.urgency.upper()}

{state.moderated_text}

📍 Area: {state.area}
📊 Similar complaints nearby: {nearby}
💬 Sentiment: {state.sentiment.title()}

ℹ️ {state.party_context}
"""
        return card

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Applying diplomatic filter")

        # Remove political attacks from text
        filtered = self.remove_political_attacks(state.moderated_text)
        state.moderated_text = filtered
        self.log(state, "Political attacks removed")

        # Generate public card
        state.public_card = self.generate_public_card(state)
        self.log(state, "Public card generated", state.public_card[:80])

        return state
