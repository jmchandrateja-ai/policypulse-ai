from datetime import datetime
from agents.state import AgentState
from agents.moderation_agent import ModerationAgent
from agents.transcription_agent import TranscriptionAgent
from agents.issue_classifier_agent import IssueClassifierAgent
from agents.sentiment_agent import SentimentAgent
from agents.topic_agent import TopicAgent
from agents.report_agent import ReportAgent


class Orchestrator:

    def __init__(self):
        self.moderation    = ModerationAgent()
        self.transcription = TranscriptionAgent()
        self.classifier    = IssueClassifierAgent()
        self.sentiment     = SentimentAgent()
        self.topic         = TopicAgent()
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
        state = self.topic.process(state)
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
        "GST rates have increased and small businesses are suffering badly",
        "Government hospital has no medicine and doctors are absent",
        "Road potholes everywhere and no repairs done for months",
        "School fees increased 40 percent and poor families cannot afford",
        "No bus service after 9pm in our area and women feel unsafe",
        "Water supply has been cut for two weeks with no explanation",
        "The bridge near our colony is dangerous and needs immediate repair",
    ]

    for complaint in complaints:
        print(f"\nProcessing: {complaint[:50]}...")
        state = orchestrator.run(complaint)

    print("\n" + "="*60)
    print("TOPIC CLUSTERING RESULTS")
    print("="*60)
    if state.topics:
        for t in state.topics:
            print(f"Topic {t['topic_id']}: {t['label']} ({t['count']} complaints)")
    else:
        print("Not enough data for topic modeling yet")
