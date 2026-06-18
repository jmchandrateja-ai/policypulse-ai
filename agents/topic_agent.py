from agents.base_agent import BaseAgent
from agents.state import AgentState
from typing import List
import json
import os

class TopicAgent(BaseAgent):

    def __init__(self):
        super().__init__("Topic Clustering Agent")
        self.complaints = []
        self.model = None
        self.fitted = False

    def add_complaint(self, text: str):
        self.complaints.append(text)

    def fit_model(self):
        if len(self.complaints) < 5:
            return False
        try:
            from bertopic import BERTopic
            self.model = BERTopic(
                language="english",
                min_topic_size=2,
                verbose=False
            )
            topics, probs = self.model.fit_transform(self.complaints)
            self.fitted = True
            return True
        except Exception as e:
            print(f"  BERTopic error: {e}")
            return False

    def get_topics(self) -> List[dict]:
        if not self.fitted or not self.model:
            return []
        topic_info = self.model.get_topic_info()
        topics = []
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                continue
            topics.append({
                "topic_id": int(row["Topic"]),
                "count": int(row["Count"]),
                "label": str(row["Name"]),
            })
        return topics[:5]

    def process(self, state: AgentState) -> AgentState:
        self.log(state, "Starting topic clustering")

        text = state.transcript if state.transcript else state.moderated_text
        self.add_complaint(text)

        self.log(state, f"Corpus size: {len(self.complaints)} complaints")

        if len(self.complaints) >= 5:
            self.log(state, "Fitting BERTopic model")
            success = self.fit_model()
            if success:
                topics = self.get_topics()
                state.topics = topics
                self.log(state, "Topics identified",
                         f"{len(topics)} topics found")
                for t in topics:
                    self.log(state, f"Topic {t['topic_id']}",
                             f"{t['label']} ({t['count']} complaints)")
            else:
                state.topics = []
                self.log(state, "Topic modeling skipped — not enough varied data")
        else:
            state.topics = []
            self.log(state, "Collecting complaints",
                     f"Need {5 - len(self.complaints)} more for topic modeling")

        return state
