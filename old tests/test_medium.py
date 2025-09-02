import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agent import Agent
from message import Message
from puzzle import Puzzle
from medium import Medium

def test_message_creation():
    message = Message(
        sender_id="agent1",
        receiver_id="agent2",
        symbols=[1, 2, 3, 64, 999]
    )
    assert message.sender_id == "agent1"
    assert message.receiver_id == "agent2"
    assert message.symbols == [1, 2, 3, 64, 999]

def test_invalid_message_symbols():
    with pytest.raises(ValueError):
        Message(
            sender_id="agent1",
            receiver_id="agent2",
            symbols=[1000]  # Invalid symbol
        )

def test_medium():
    medium = Medium()
    agent1 = Agent("agent1")
    agent2 = Agent("agent2")
    
    # Register agents
    medium.register_agent(agent1)
    medium.register_agent(agent2)
    
    # Test message passing
    message = Message(
        sender_id="agent1",
        receiver_id="agent2",
        symbols=[1, 2, 3]
    )
    
    # Initially no pending messages
    assert not medium.has_pending_messages("agent2")
    
    # Send message
    medium.send_message(message)
    
    # Now agent2 should have a pending message
    assert medium.has_pending_messages("agent2")
    
    # Receive message
    received = medium.receive_message("agent2")
    assert received is not None
    assert received.sender_id == "agent1"
    assert received.symbols == [1, 2, 3]
    
    # Queue should be empty again
    assert not medium.has_pending_messages("agent2")

def test_unregistered_agent():
    medium = Medium()
    
    with pytest.raises(ValueError):
        medium.receive_message("nonexistent")