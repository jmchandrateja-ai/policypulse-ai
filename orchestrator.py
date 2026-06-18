from datetime import datetime
from agents.state import AgentState
from agents.moderation_agent import ModerationAgent
from agents.transcription_agent import TranscriptionAgent
from agents.issue_classifier_agent import IssueClassifierAgent
from agents.sentiment_agent import SentimentAgent
from agents.topic_agent import TopicAgent
from agents.voice_reader_agent import VoiceReaderAgent
from agents.confirmation_agent import ConfirmationAgent
from agents.report_agent import ReportAgent


class Orchestrator:

    def __init__(self):
        self.moderation    = ModerationAgent()
        self.transcription = TranscriptionAgent()
        self.classifier    = IssueClassifierAgent()
        self.sentiment     = SentimentAgent()
        self.topic         = TopicAgent()
        self.voice_reader  = VoiceReaderAgent()
        self.confirmation  = ConfirmationAgent()
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
        state = self.voice_reader.process(state)
        state = self.confirmation.process(state)
        state = self.report.process(state)

        state.status       = "complete"
        state.completed_at = datetime.now().strftime("%H:%M:%S")
        state.log("Orchestrator", "Pipeline complete",
                  f"Domain: {state.domain} | Urgency: {state.urgency}")

        return state
