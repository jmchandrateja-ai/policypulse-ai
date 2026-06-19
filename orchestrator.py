from datetime import datetime
from agents.state import AgentState
from agents.intake_agent import IntakeAgent
from agents.intelligence_agent import IntelligenceAgent
from agents.geospatial_agent import GeospatialAgent
from agents.diplomatic_agent import DiplomaticAgent
from agents.secretary_agent import SecretaryAgent


class Orchestrator:

    def __init__(self):
        self.intake       = IntakeAgent()
        self.intelligence = IntelligenceAgent()
        self.geospatial   = GeospatialAgent()
        self.diplomatic   = DiplomaticAgent()
        self.secretary    = SecretaryAgent()

    def run(self, input_text: str = "",
            audio_path: str = None,
            media_path: str = None,
            latitude: float = None,
            longitude: float = None) -> AgentState:

        state = AgentState()
        state.input_text  = input_text
        state.audio_path  = audio_path
        state.media_path  = media_path
        state.status      = "running"
        state.started_at  = datetime.now().strftime("%H:%M:%S")

        if latitude:  state.latitude  = latitude
        if longitude: state.longitude = longitude

        state.log("Orchestrator", "CivicLens AI pipeline started",
                  f"Input: {input_text[:60]}")

        state = self.intake.process(state)
        state = self.intelligence.process(state)
        state = self.geospatial.process(state)
        state = self.diplomatic.process(state)
        state = self.secretary.process(state)

        state.status       = "complete"
        state.completed_at = datetime.now().strftime("%H:%M:%S")
        state.log("Orchestrator", "Pipeline complete",
                  f"Domain: {state.domain} | "
                  f"Area: {state.area} | "
                  f"Urgency: {state.urgency}")

        return state


if __name__ == "__main__":
    o = Orchestrator()

    print("\nTEST 1 — Infrastructure complaint")
    state = o.run("The road has potholes everywhere and nobody is fixing them")
    print(state.public_card)
    print(state.legislator_brief)

    print("\nTEST 2 — Healthcare urgent")
    state = o.run("Government hospital has no medicine. This is an emergency!")
    print(state.public_card)
