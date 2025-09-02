import torch
import numpy as np
import torch.nn.functional as F
import sys
import traceback

# Import key components
from agent import Agent
from trainer import CommunicationTrainer
from puzzle import Puzzle

def generate_test_puzzle(size=(10, 10), num_symbols=10):
    """Generate a simple test puzzle with random symbols."""
    puzzle_grid = torch.randint(0, num_symbols, size=(1, size[0], size[1]), dtype=torch.long)
    return puzzle_grid

def check_tensor_properties(tensor, name):
    """Check tensor for any numerical issues."""
    print(f"\nChecking {name}:")
    if tensor is None:
        print(f"  {name} is None!")
        return False
    
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item()}")
    print(f"  Max: {tensor.max().item()}")
    print(f"  Mean: {tensor.mean().item()}")
    print(f"  Contains NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Contains Inf: {torch.isinf(tensor).any().item()}")
    
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())

def test_agent_encoding(agent, puzzle):
    """Test encoding functionality."""
    print("\n--- Testing Agent Encoding ---")
    try:
        symbols, symbol_logits, length_stats = agent.encode_puzzle_to_message(
            puzzle, 
            temperature=1.0, 
            initial_phase=False
        )
        
        # Check all components
        checks = [
            check_tensor_properties(symbols, "Symbols"),
            check_tensor_properties(symbol_logits, "Symbol Logits")
        ]
        
        print("\nLength Statistics:")
        print(f"  Total Length: {length_stats['total_length']}")
        print(f"  Non-zero Symbols: {length_stats['nonzero_symbols']}")
        
        return all(checks)
    except Exception as e:
        print(f"Error in encoding: {e}")
        traceback.print_exc()
        return False

def test_agent_decoding(agent, symbols, target_size):
    """Test decoding functionality."""
    print("\n--- Testing Agent Decoding ---")
    try:
        grid, grid_logits, intermediate_logits, confidence_scores = agent.decode_message_to_puzzle(
            symbols, 
            target_size=target_size, 
            temperature=1.0, 
            hard=True
        )
        
        # Check all components
        checks = [
            check_tensor_properties(grid, "Decoded Grid"),
            check_tensor_properties(grid_logits, "Grid Logits")
        ]
        
        # Check intermediate logits
        print("\nIntermediate Logits:")
        for i, inter_logit in enumerate(intermediate_logits):
            print(f"  Intermediate {i}:")
            check_tensor_properties(inter_logit, f"Intermediate Logits {i}")
        
        # Check confidence scores
        print("\nConfidence Scores:")
        for i, conf_score in enumerate(confidence_scores):
            print(f"  Confidence {i}:")
            check_tensor_properties(conf_score, f"Confidence Score {i}")
        
        return all(checks)
    except Exception as e:
        print(f"Error in decoding: {e}")
        traceback.print_exc()
        return False

def test_embedding_system(agent, puzzle):
    """Test embedding system functionality."""
    print("\n--- Testing Embedding System ---")
    try:
        # Embed the puzzle
        puzzle_emb = agent.embedding_system.embed_puzzle(puzzle)
        return check_tensor_properties(puzzle_emb, "Puzzle Embedding")
    except Exception as e:
        print(f"Error in embedding: {e}")
        traceback.print_exc()
        return False

def comprehensive_agent_test(agent_type='sender'):
    """Comprehensive test of an agent's functionality."""
    print(f"\n{'='*20} TESTING {agent_type.upper()} AGENT {'='*20}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create an agent
    agent = Agent(
        agent_id=agent_type,
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,
        puzzle_symbols=10,
        max_seq_length=1,
        sender_scale=25.0 if agent_type == 'sender' else None
    )
    
    # Generate test puzzle
    puzzle = generate_test_puzzle()
    
    # Run comprehensive tests
    tests = [
        ("Embedding System", test_embedding_system(agent, puzzle)),
        ("Encoding", test_agent_encoding(agent, puzzle)),
    ]
    
    # If encoding succeeds, test decoding
    if tests[-1][1]:
        # Reuse the symbols from encoding
        symbols, _, _ = agent.encode_puzzle_to_message(puzzle, temperature=1.0)
        decoding_test = test_agent_decoding(
            agent, 
            symbols, 
            target_size=(puzzle.size(1), puzzle.size(2))
        )
        tests.append(("Decoding", decoding_test))
    
    # Print overall results
    print("\n--- TEST SUMMARY ---")
    all_passed = all(test[1] for test in tests)
    for name, passed in tests:
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")
    
    return all_passed

def test_communication_trainer():
    """Test the communication trainer."""
    print("\n{'='*20} TESTING COMMUNICATION TRAINER {'='*20}")
    
    # Create two agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,
        puzzle_symbols=10,
        max_seq_length=1,
        sender_scale=25.0
    )
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,
        puzzle_symbols=10,
        max_seq_length=1
    )
    
    # Create trainer
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=1e-4
    )
    
    # Generate test puzzle
    puzzle = generate_test_puzzle()
    
    try:
        # Attempt bidirectional training step
        metrics = trainer.train_bidirectional_step(
            puzzle, 
            num_exchanges=1,
            temperature=1.0,
            initial_phase=True
        )
        
        # Check metrics
        print("\nMetrics Analysis:")
        for i, metric in enumerate(metrics):
            print(f"\nMetric {i}:")
            for key, value in metric.items():
                print(f"  {key}: {value}")
                if isinstance(value, torch.Tensor):
                    check_tensor_properties(value, key)
                elif isinstance(value, float):
                    print(f"    Is NaN: {np.isnan(value)}")
        
        return True
    except Exception as e:
        print(f"Error in trainer test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostic tests."""
    print("Starting Comprehensive Diagnostic Tests...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run tests
    tests = [
        ("Sender Agent Test", comprehensive_agent_test('sender')),
        ("Receiver Agent Test", comprehensive_agent_test('receiver')),
        ("Communication Trainer Test", test_communication_trainer())
    ]
    
    # Print final summary
    print("\n{'='*30} FINAL TEST SUMMARY {'='*30}")
    all_passed = all(test[1] for test in tests)
    for name, passed in tests:
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()