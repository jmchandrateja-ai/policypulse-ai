from datetime import datetime
from agents.state import AgentState
from agents.moderation_agent import ModerationAgent
from agents.transcription_agent import TranscriptionAgent
from agents.issue_classifier_agent import IssueClassifierAgent
from agents.sentiment_agent import SentimentAgent
from agents.report_agent import ReportAgent


class Orchestrator:

    def __init__(self):
        self.moderation    = ModerationAgent()
        self.transcription = TranscriptionAgent()
        self.classifier    = IssueClassifierAgent()
        self.sentiment     = SentimentAgent()
        self.report        = ReportAgent()

    def run(self, input_text: str, audio_path: str = None) -> AgentState:
        state = AgentState()
        state.input_text = input_text
        state.audio_path = audio_path
        state.status     = "running"
        state.started_at = datetime.now().strftime("%H:%M:%S")

        state.log("Orchestrator", "Pipeline started",
                  f"Input: {input_text[:60]}")

        state = self.moderation.process(state)
        state = self.transcription.process(state)
        state = self.classifier.process(state)

        if state.routing_decision == "clarify":
            state.status = "needs_clarification"
            state.log("Orchestrator", "Pipeline paused",
                      "Low confidence — clarification needed")
            return state

        state = self.sentiment.process(state)
        state = self.report.process(state)

        state.status       = "complete"
        state.completed_at = datetime.now().strftime("%H:%M:%S")
        state.log("Orchestrator", "Pipeline complete",
                  f"Domain: {state.domain} | Urgency: {state.urgency}")

        return state


if __name__ == "__main__":
    orchestrator = Orchestrator()

    complaints = [
        "The new property tax is too high and unaffordable for retired citizens",
        "This stupid government is ruining the roads with potholes everywhere",
        "The government hospital has no medicine and doctors are absent. This is urgent!",
        "I support the new metro expansion. It will help daily commuters greatly.",
    ]

    for complaint in complaints:
        print("\n" + "="*60)
        print(f"INPUT: {complaint}")
        print("="*60)
        state = orchestrator.run(complaint)
        print(state.legislator_brief)
