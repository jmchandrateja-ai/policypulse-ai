from agents.base_agent import BaseAgent
from agents.state import AgentState
from datetime import datetime


class ReportAgent(BaseAgent):

    def __init__(self):
        super().__init__("Report Agent")

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Generating legislator intelligence brief")

        if state.urgency == "high":
            action = "IMMEDIATE ATTENTION REQUIRED"
        elif state.urgency == "medium":
            action = "REVIEW RECOMMENDED"
        else:
            action = "LOGGED — No immediate action required"

        if state.stance == "oppose":
            stance_label = "Citizens OPPOSE this policy"
        elif state.stance == "support":
            stance_label = "Citizens SUPPORT this policy"
        else:
            stance_label = "Citizens are NEUTRAL on this policy"

        abusive = "Yes — core issue extracted" if state.is_abusive else "No"

        lines = [
            "",
            "=" * 58,
            "     POLICYPULSE AI — LEGISLATOR BRIEF",
            "=" * 58,
            "",
            "Date       : " + datetime.now().strftime("%d %B %Y, %H:%M"),
            "Urgency    : " + state.urgency.upper(),
            "",
            "CITIZEN COMPLAINT",
            "-" * 40,
            "Original   : " + state.input_text,
            "Cleaned    : " + state.moderated_text,
            "Abusive    : " + abusive,
            "",
            "POLICY INTELLIGENCE",
            "-" * 40,
            "Domain     : " + state.domain.upper(),
            "Confidence : " + str(round(state.domain_confidence * 100)) + "%",
            "Routing    : " + state.routing_decision.upper(),
            "",
            "SENTIMENT ANALYSIS",
            "-" * 40,
            "Sentiment  : " + state.sentiment.upper(),
            "Stance     : " + state.stance.upper(),
            "Summary    : " + stance_label,
            "",
            "ACTION",
            "-" * 40,
            action,
            "",
            "=" * 58,
        ]

        state.legislator_brief = "\n".join(lines)
        self.log(state, "Brief generated",
                 "Domain: " + state.domain + " | Urgency: " + state.urgency)

        return state
