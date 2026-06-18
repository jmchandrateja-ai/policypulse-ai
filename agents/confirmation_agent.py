from agents.base_agent import BaseAgent
from agents.state import AgentState
import os
import time
import random

class ConfirmationAgent(BaseAgent):

    def __init__(self):
        super().__init__("Confirmation Call Agent")
        os.makedirs("outputs/audio", exist_ok=True)

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Generating confirmation message")
        ref = f"PP{random.randint(10000, 99999)}"
        domain = state.domain.title() if state.domain else "General"

        if state.urgency == "high":
            action = "This has been marked HIGH PRIORITY and will be reviewed within 24 hours."
        elif state.urgency == "medium":
            action = "This will be reviewed within 3 working days."
        else:
            action = "This has been logged and will be reviewed within 7 working days."

        message = (
            f"Namaste. This is PolicyPulse AI. "
            f"Your complaint regarding {domain} policy has been received. "
            f"Your reference number is {ref}. "
            f"{action} "
            f"Thank you. Your voice matters."
        )

        self.log(state, "Message created", message[:80])

        try:
            from gtts import gTTS
            filename = f"confirmation_{int(time.time())}"
            path = f"outputs/audio/{filename}.mp3"
            tts = gTTS(text=message, lang="en", slow=False)
            tts.save(path)
            state.confirmation_audio = path
            self.log(state, "Confirmation audio ready", path)
        except Exception as e:
            state.confirmation_audio = None
            self.log(state, "Audio failed", str(e))

        return state
