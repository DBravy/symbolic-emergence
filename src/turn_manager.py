from typing import List, Optional
from agent import Agent
from medium import Medium
from puzzle import Puzzle

class TurnManager:
    def __init__(self, medium: Medium, puzzle: Puzzle):
        self.medium = medium
        self.puzzle = puzzle
        self.agents: List[Agent] = []
        self.current_turn_idx = 0
        
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the turn rotation"""
        if agent not in self.agents:
            self.agents.append(agent)
            self.medium.register_agent(agent)
    
    def get_current_agent(self) -> Optional[Agent]:
        """Get the agent whose turn it currently is"""
        if not self.agents:
            return None
        return self.agents[self.current_turn_idx]
    
    def advance_turn(self) -> None:
        """Move to the next agent's turn"""
        if self.agents:
            self.current_turn_idx = (self.current_turn_idx + 1) % len(self.agents)
            
    def is_agent_turn(self, agent_id: str) -> bool:
        """Check if it's the specified agent's turn"""
        current_agent = self.get_current_agent()
        return current_agent is not None and current_agent.get_id() == agent_id