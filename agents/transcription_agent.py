from agents.base_agent import BaseAgent
from agents.state import AgentState


class TranscriptionAgent(BaseAgent):

    def __init__(self):
        super().__init__("Voice Transcription Agent")
        self.model = None

    def load_model(self):
        import whisper
        self.model = whisper.load_model("base")
        return self.model

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting transcription")

        # If no audio, use input_text directly
        if not state.audio_path:
            self.log(state, "No audio file — using text input directly")
            if not state.transcript:
                state.transcript = state.moderated_text
            return state

        # Load model if not loaded
        if not self.model:
            self.log(state, "Loading Whisper model")
            self.load_model()

        # Transcribe
        self.log(state, "Transcribing audio", state.audio_path)
        result = self.model.transcribe(state.audio_path)
        state.transcript = result["text"].strip()
        state.input_language = result.get("language", "en")

        self.log(state, "Transcription complete",
                 f"Language: {state.input_language} | Text: {state.transcript[:80]}")

        return state
