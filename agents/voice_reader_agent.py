from agents.base_agent import BaseAgent
from agents.state import AgentState
import os
import time

class VoiceReaderAgent(BaseAgent):

    def __init__(self):
        super().__init__("Policy Voice Reader Agent")
        os.makedirs("outputs/audio", exist_ok=True)

    def simplify_text(self, text):
        replacements = {
            "shall be liable": "must pay",
            "in accordance with": "following",
            "with respect to": "about",
            "prior to": "before",
            "subsequent to": "after",
            "in the event of": "if",
            "is required to": "must",
            "is entitled to": "can",
        }
        for legal, plain in replacements.items():
            text = text.replace(legal, plain)
        return text.strip()

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting policy voice reading")
        text = state.moderated_text
        if not text.strip():
            self.log(state, "No text to read")
            return state
        simplified = self.simplify_text(text)
        self.log(state, "Text simplified", simplified[:80])
        try:
            from gtts import gTTS
            filename = f"policy_{int(time.time())}"
            path = f"outputs/audio/{filename}.mp3"
            tts = gTTS(text=simplified, lang="en", slow=False)
            tts.save(path)
            self.log(state, "Audio generated", path)
        except Exception as e:
            self.log(state, "Audio failed", str(e))
        return state
