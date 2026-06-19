from agents.base_agent import BaseAgent
from agents.state import AgentState
import sqlite3
import os
import time
import random
from datetime import datetime


class SecretaryAgent(BaseAgent):

    def __init__(self):
        super().__init__("AI Secretary Agent")
        os.makedirs("outputs/audio", exist_ok=True)

    def save_complaint(self, state: AgentState) -> int:
        try:
            conn = sqlite3.connect("data/complaints.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO complaints (
                    timestamp, input_type, original_text, cleaned_text,
                    domain, sentiment, stance, urgency, confidence,
                    routing, latitude, longitude, area, ward,
                    media_path, party_context, legislator_brief,
                    confirmation_audio, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(),
                "voice" if state.audio_path else "text",
                state.input_text,
                state.moderated_text,
                state.domain,
                state.sentiment,
                state.stance,
                state.urgency,
                state.domain_confidence,
                state.routing_decision,
                state.latitude,
                state.longitude,
                state.area,
                state.ward,
                getattr(state, 'media_path', None),
                state.party_context,
                state.legislator_brief,
                state.confirmation_audio,
                "open"
            ))
            complaint_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return complaint_id
        except Exception as e:
            print(f"DB error: {e}")
            return 0

    def generate_brief(self, state: AgentState) -> str:
        nearby = state.nearby_count
        lines = [
            "=" * 55,
            "     CIVICLE NS AI — LEGISLATOR BRIEF",
            "=" * 55,
            f"Date       : {datetime.now().strftime('%d %B %Y, %H:%M')}",
            f"Urgency    : {state.urgency.upper()}",
            f"Area       : {state.area}",
            "",
            "CITIZEN COMPLAINT",
            "-" * 40,
            f"Text       : {state.moderated_text}",
            f"Abusive    : {'Yes — cleaned' if state.is_abusive else 'No'}",
            "",
            "INTELLIGENCE SUMMARY",
            "-" * 40,
            f"Domain     : {state.domain.upper()}",
            f"Sentiment  : {state.sentiment.upper()}",
            f"Stance     : {state.stance.upper()}",
            f"Confidence : {state.domain_confidence:.0%}",
            f"Routing    : {state.routing_decision.upper()}",
            "",
            "GEOSPATIAL CONTEXT",
            "-" * 40,
            f"Location   : {state.latitude:.4f}, {state.longitude:.4f}",
            f"Area       : {state.area}",
            f"Similar complaints nearby: {nearby}",
            "",
            "AUTHORITY CONTEXT",
            "-" * 40,
            state.party_context,
            "",
            "ACTION REQUIRED",
            "-" * 40,
        ]

        if state.urgency == "high":
            lines.append("🔴 IMMEDIATE ATTENTION — High urgency complaint")
        elif state.urgency == "medium":
            lines.append("🟡 REVIEW RECOMMENDED — Add to weekly report")
        else:
            lines.append("🟢 LOGGED — No immediate action required")

        if nearby >= 5:
            lines.append(f"⚠️  HOTSPOT ALERT — {nearby} similar complaints in this area")

        lines.append("=" * 55)
        return "\n".join(lines)

    def generate_confirmation(self, state: AgentState, ref: str) -> str:
        domain  = state.domain.title()
        urgency = state.urgency

        if urgency == "high":
            action = "This has been marked HIGH PRIORITY and will be reviewed within 24 hours."
        elif urgency == "medium":
            action = "This will be reviewed within 3 working days."
        else:
            action = "This has been logged and will be reviewed within 7 working days."

        return (
            f"Namaste. This is CivicLens AI, your civic services assistant. "
            f"Your complaint regarding {domain} in {state.area} has been received. "
            f"Your reference number is {ref}. "
            f"{action} "
            f"Your complaint is now visible on the public civic feed. "
            f"Thank you for participating. Your voice matters."
        )

    def text_to_speech(self, text: str, filename: str) -> str:
        try:
            from gtts import gTTS
            path = f"outputs/audio/{filename}.mp3"
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(path)
            return path
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "AI Secretary processing")

        # Generate reference number
        ref = f"CL{random.randint(10000, 99999)}"

        # Generate legislator brief
        state.legislator_brief = self.generate_brief(state)

        # Save to database
        complaint_id = self.save_complaint(state)
        state.complaint_id = complaint_id
        self.log(state, "Complaint saved to database", f"ID: {complaint_id}")

        # Generate confirmation message
        confirmation_text = self.generate_confirmation(state, ref)

        # Convert to speech
        filename = f"confirmation_{int(time.time())}"
        audio_path = self.text_to_speech(confirmation_text, filename)
        if audio_path:
            state.confirmation_audio = audio_path
            self.log(state, "Confirmation audio ready", audio_path)

        self.log(state, "Secretary complete",
                 f"Ref: {ref} | Area: {state.area} | Urgency: {state.urgency}")

        return state
