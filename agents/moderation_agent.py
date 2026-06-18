from agents.base_agent import BaseAgent
from agents.state import AgentState

ABUSIVE_KEYWORDS = [
    "idiot", "stupid", "fool", "bastard", "shut up",
    "bakwaas", "bewakoof", "gadha", "saala", "chutiya",
    "ullu", "haramkhor", "nikamma"
]

class ModerationAgent(BaseAgent):

    def __init__(self):
        super().__init__("Content Moderation Agent")

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting content moderation")

        text = state.input_text.lower()

        found = [w for w in ABUSIVE_KEYWORDS if w in text]

        if found:
            state.is_abusive = True
            self.log(state, "Abusive language detected", f"Words: {found}")
            clean = state.input_text
            for w in found:
                clean = clean.replace(w, "").replace(w.capitalize(), "")
            state.moderated_text = " ".join(clean.split())
            self.log(state, "Core issue extracted", state.moderated_text)
        else:
            state.is_abusive = False
            state.moderated_text = state.input_text
            self.log(state, "No abusive content found")

        return state
