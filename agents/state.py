from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class AgentState:

    # Input
    input_text: str = ""
    input_language: str = ""
    audio_path: Optional[str] = None

    # Agent Outputs
    moderated_text: str = ""
    is_abusive: bool = False
    transcript: str = ""

    domain: str = ""
    issue_type: str = ""
    domain_confidence: float = 0.0

    sentiment: str = ""
    stance: str = ""
    urgency: str = ""

    topics: List[Dict] = field(default_factory=list)
    policy_provisions: List[Dict] = field(default_factory=list)

    confirmation_audio: Optional[str] = None
    legislator_brief: str = ""

    routing_decision: str = ""

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
