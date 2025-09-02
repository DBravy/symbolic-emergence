import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agent import Agent
from medium import Medium
from puzzle import Puzzle
from turn_manager import TurnManager

def test_turn_manager():
    # Create a simple puzzle
    input_grid = np.array([[0, 1], [2, 3]])
    output_grid = np.array([[1, 2], [3, 4]])
    puzzle = Puzzle.from_single_example(input_grid, output_grid)
    
    # Create medium and agents
    medium = Medium()
    agent1 = Agent("agent1")
    agent2 = Agent("agent2")
    
    # Set up turn manager
    manager = TurnManager(medium, puzzle)
    manager.add_agent(agent1)
    manager.add_agent(agent2)
    
    # Test turn rotation
    assert manager.is_agent_turn("agent1")
    assert not manager.is_agent_turn("agent2")
    
    manager.advance_turn()
    assert not manager.is_agent_turn("agent1")
    assert manager.is_agent_turn("agent2")
    
    manager.advance_turn()
    assert manager.is_agent_turn("agent1")