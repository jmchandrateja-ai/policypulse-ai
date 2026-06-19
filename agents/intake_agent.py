from agents.base_agent import BaseAgent
from agents.state import AgentState
import os
import re

ABUSIVE_KEYWORDS = [
    "idiot", "stupid", "fool", "bastard", "shut up",
    "bakwaas", "bewakoof", "gadha", "saala", "chutiya",
    "ullu", "haramkhor", "nikamma"
]

class IntakeAgent(BaseAgent):

    def __init__(self):
        super().__init__("Intake Agent")
        self.whisper_model = None

    def load_whisper(self):
        import whisper
        self.whisper_model = whisper.load_model("base")

    def moderate(self, text: str):
        text_lower = text.lower()
        found = [w for w in ABUSIVE_KEYWORDS if w in text_lower]
        if found:
            clean = text
            for w in found:
                clean = clean.replace(w, "").replace(w.capitalize(), "")
            return True, " ".join(clean.split())
        return False, text

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper_model:
            self.load_whisper()
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    def extract_geotag(self, image_path: str):
        try:
            import piexif
            from PIL import Image
            img = Image.open(image_path)
            exif_data = piexif.load(img.info.get("exif", b""))
            gps = exif_data.get("GPS", {})
            if not gps:
                return None, None

            def to_decimal(vals):
                d, m, s = vals
                return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

            lat = to_decimal(gps.get(piexif.GPSIFD.GPSLatitude, [(0,1),(0,1),(0,1)]))
            lon = to_decimal(gps.get(piexif.GPSIFD.GPSLongitude, [(0,1),(0,1),(0,1)]))
            return lat, lon
        except Exception as e:
            self.log_print(f"Geotag extraction error: {e}")
            return None, None

    def log_print(self, msg):
        print(f"  [{self.name}] {msg}")

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Intake Agent starting")

        # Step 1 - transcribe if audio
        if state.audio_path and not state.input_text.strip():
            self.log(state, "Transcribing audio input")
            state.input_text = self.transcribe(state.audio_path)
            state.input_language = "en"
            self.log(state, "Transcription complete", state.input_text[:80])

        # Step 2 - extract geotag if image
        if hasattr(state, 'media_path') and state.media_path:
            ext = os.path.splitext(state.media_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                lat, lon = self.extract_geotag(state.media_path)
                if lat and lon:
                    state.latitude = lat
                    state.longitude = lon
                    self.log(state, "Geotag extracted",
                             f"Lat: {lat:.4f} | Lon: {lon:.4f}")
                else:
                    # Default to Bangalore if no geotag
                    state.latitude = 12.9716
                    state.longitude = 77.5946
                    self.log(state, "No geotag found — defaulting to Bangalore")
            else:
                state.latitude = 12.9716
                state.longitude = 77.5946
        elif not hasattr(state, 'latitude') or not state.latitude:
            state.latitude = 12.9716
            state.longitude = 77.5946

        # Step 3 - moderate text
        is_abusive, clean_text = self.moderate(state.input_text)
        state.is_abusive = is_abusive
        state.moderated_text = clean_text

        if is_abusive:
            self.log(state, "Abusive content detected and cleaned")
        else:
            self.log(state, "Content moderation passed")

        self.log(state, "Intake complete", state.moderated_text[:80])
        return state
