from datetime import datetime
from agents.state import AgentState
from agents.moderation_agent import ModerationAgent
from agents.transcription_agent import TranscriptionAgent
from agents.issue_classifier_agent import IssueClassifierAgent
from agents.sentiment_agent import SentimentAgent


class Orchestrator:

    def __init__(self):
        self.moderation   = ModerationAgent()
        self.transcription = TranscriptionAgent()
        self.classifier   = IssueClassifierAgent()
        self.sentiment    = SentimentAgent()

    def run(self, input_text: str, audio_path: str = None) -> AgentState:
        state = AgentState()
        state.input_text = input_text
        state.audio_path = audio_path
        state.status     = "running"
        state.started_at = datetime.now().strftime("%H:%M:%S")

        state.log("Orchestrator", "Pipeline started",
                  f"Input: {input_text[:60]}")

        # Run agents in sequence
        state = self.moderation.process(state)

        state = self.transcription.process(state)

        state = self.classifier.process(state)

        # Stop if clarification needed
        if state.routing_decision == "clarify":
            state.status = "needs_clarification"
            state.log("Orchestrator", "Pipeline paused",
                      "Low confidence — clarification needed")
            return state

        state = self.sentiment.process(state)

        state.status       = "complete"
        state.completed_at = datetime.now().strftime("%H:%M:%S")
        state.log("Orchestrator", "Pipeline complete",
                  f"Domain: {state.domain} | "
                  f"Sentiment: {state.sentiment} | "
                  f"Stance: {state.stance}")

        return state


def print_report(state: AgentState):
    print("\n" + "="*50)
    print("POLICYPULSE AI — COMPLAINT ANALYSIS REPORT")
    print("="*50)
    print(f"Input:      {state.input_text}")
    print(f"Cleaned:    {state.moderated_text}")
    print(f"Domain:     {state.domain}")
    print(f"Confidence: {state.domain_confidence:.2f}")
    print(f"Routing:    {state.routing_decision}")
    print(f"Sentiment:  {state.sentiment}")
    print(f"Stance:     {state.stance}")
    print(f"Urgency:    {state.urgency}")
    print(f"Status:     {state.status}")
    print("-"*50)
    print("THOUGHT LOG:")
    for entry in state.thought_log:
        print(f"  [{entry['time']}] {entry['agent']}: {entry['action']}")
        if entry['detail']:
            print(f"    -> {entry['detail']}")
    print("="*50)


if __name__ == "__main__":
    orchestrator = Orchestrator()

    # Test 1 — Normal complaint
    print("\nTEST 1 — Normal complaint")
    state = orchestrator.run(
        "The new property tax is too high and unaffordable for retired citizens"
    )
    print_report(state)

    # Test 2 — Abusive complaint
    print("\nTEST 2 — Abusive complaint")
    state = orchestrator.run(
        "This stupid government is ruining the roads with potholes everywhere"
    )
    print_report(state)

    # Test 3 — Healthcare complaint
    print("\nTEST 3 — Healthcare complaint")
    state = orchestrator.run(
        "The government hospital has no medicine and doctors are absent"
    )
    print_report(state)
