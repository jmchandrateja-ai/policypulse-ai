from agents.base_agent import BaseAgent
from agents.state import AgentState
import sqlite3

# Bangalore ward boundaries approximation
BANGALORE_WARDS = {
    "Whitefield":      (12.9698, 77.7500),
    "Koramangala":     (12.9352, 77.6245),
    "Indiranagar":     (12.9784, 77.6408),
    "Jayanagar":       (12.9308, 77.5838),
    "Hebbal":          (13.0358, 77.5970),
    "Electronic City": (12.8458, 77.6603),
    "Marathahalli":    (12.9591, 77.6974),
    "Rajajinagar":     (12.9919, 77.5560),
    "Malleshwaram":    (13.0035, 77.5714),
    "BTM Layout":      (12.9166, 77.6101),
    "Bangalore":       (12.9716, 77.5946),
}

# Party context by domain — diplomatic framing
PARTY_CONTEXT = {
    "infrastructure": "BBMP (Bruhat Bengaluru Mahanagara Palike) is responsible for road and infrastructure maintenance in Bangalore.",
    "healthcare":     "Karnataka Department of Health and Family Welfare oversees public healthcare services.",
    "education":      "Karnataka Department of Primary and Secondary Education manages government schools and fee structures.",
    "taxation":       "Karnataka Commercial Taxes Department and BBMP handle local tax administration.",
    "housing":        "Karnataka Housing Board and BBMP manage affordable housing and slum redevelopment.",
    "environment":    "Karnataka State Pollution Control Board and BBMP handle environmental compliance.",
    "transportation": "BMTC (Bangalore Metropolitan Transport Corporation) and BMRCL manage public transport.",
    "other":          "Relevant municipal and state authorities handle this category.",
}


class GeospatialAgent(BaseAgent):

    def __init__(self):
        super().__init__("Geospatial Agent")

    def find_nearest_area(self, lat: float, lon: float) -> str:
        min_dist = float("inf")
        nearest  = "Bangalore"
        for area, (alat, alon) in BANGALORE_WARDS.items():
            dist = ((lat - alat)**2 + (lon - alon)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest  = area
        return nearest

    def get_nearby_count(self, lat: float, lon: float, domain: str) -> int:
        try:
            conn = sqlite3.connect("data/complaints.db")
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM complaints
                WHERE domain = ?
                AND ABS(latitude - ?) < 0.05
                AND ABS(longitude - ?) < 0.05
            """, (domain, lat, lon))
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting geospatial analysis")

        lat = state.latitude or 12.9716
        lon = state.longitude or 77.5946

        # Find nearest area
        state.area = self.find_nearest_area(lat, lon)
        self.log(state, "Area identified", state.area)

        # Get nearby complaint count
        state.nearby_count = self.get_nearby_count(lat, lon, state.domain)
        self.log(state, "Nearby complaints",
                 f"{state.nearby_count} similar complaints in this area")

        # Add party context — diplomatic framing
        state.party_context = PARTY_CONTEXT.get(
            state.domain, PARTY_CONTEXT["other"]
        )
        self.log(state, "Party context added", state.party_context[:80])

        return state
