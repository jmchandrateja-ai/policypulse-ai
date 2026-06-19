from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class AgentState:

    # Input
    input_text: str = ""
    input_language: str = ""
    audio_path: Optional[str] = None
    media_path: Optional[str] = None

    # Location
    latitude: float = 12.9716
    longitude: float = 77.5946
    area: str = ""
    ward: str = ""

    # Moderation
    moderated_text: str = ""
    is_abusive: bool = False

    # Intelligence
    transcript: str = ""
    domain: str = ""
    issue_type: str = ""
    domain_confidence: float = 0.0
    sentiment: str = ""
    stance: str = ""
    urgency: str = ""
    routing_decision: str = ""

    # Clustering
    topics: List[Dict] = field(default_factory=list)
    cluster_id: Optional[int] = None
    nearby_count: int = 0

    # Party context
    party_context: str = ""

    # Output
    policy_provisions: List[Dict] = field(default_factory=list)
    confirmation_audio: Optional[str] = None
    legislator_brief: str = ""
    public_card: str = ""
    complaint_id: Optional[int] = None

    # Pipeline
    thought_log: List[Dict] = field(default_factory=list)
    status: str = "idle"
    started_at: str = ""
    completed_at: str = ""

    def log(self, agent: str, action: str, detail: str = ""):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "action": action,
            "detail": detail
        }
        self.thought_log.append(entry)
        print(f"[{entry['time']}] {agent}: {action}")
        if detail:
            print(f"  -> {detail}")
