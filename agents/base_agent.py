from abc import ABC, abstractmethod
from agents.state import AgentState


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        pass

    def log(self, state: AgentState, action: str, detail: str = ""):
        state.log(self.name, action, detail)
