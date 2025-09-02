import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import Agent
from medium import Medium
from trainer import CommunicationTrainer
import matplotlib.pyplot as plt
from typing import List, Tuple
from puzzle import Puzzle

def analyze_puzzle_distribution(puzzles: List[Puzzle]) -> dict:
    """Analyze symbol distribution across all puzzles."""
    all_symbols = []
    for puzzle in puzzles:
        all_symbols.extend(puzzle.test_input.ravel())
    
    unique, counts = np.unique(all_symbols, return_counts=True)
    return dict(zip(unique, counts))

def analyze_decoder_outputs(
    agent: Agent,
    puzzles: List[Puzzle],
    num_samples: int = 10
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Analyze decoder outputs for different inputs.
    Returns the first output and a list of differences from first output.
    """
    outputs = []
    for i in range(min(num_samples, len(puzzles))):
        puzzle = puzzles[i]
        puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long).unsqueeze(0)
        
        # Force using only one communication symbol
        symbols = torch.ones((1, 1), dtype=torch.long, device=puzzle_tensor.device) * agent.puzzle_symbols
        
        with torch.no_grad():
            output, _ = agent.decode_message_to_puzzle(
                symbols,
                target_size=(puzzle_tensor.size(1), puzzle_tensor.size(2))
            )
        outputs.append(output.cpu().numpy())
    
    # Compare all outputs to first output
    first_output = outputs[0]
    differences = [np.sum(output != first_output) for output in outputs[1:]]
    
    return first_output, differences

def run_analysis():
    # Initialize agents with minimal capacity
    agent1 = Agent(
        "agent1",
        embedding_dim=64,
        hidden_dim=128,
        num_symbols=65,  # Only one communication symbol (64 puzzle + 1 comm)
        puzzle_symbols=64,
        max_seq_length=1,
        max_grid_size=(30, 30),
        fixed_size=False
    )
    agent2 = Agent(
        "agent2",
        embedding_dim=64,
        hidden_dim=128,
        num_symbols=65,
        puzzle_symbols=64,
        max_seq_length=1,
        max_grid_size=(30, 30),
        fixed_size=False
    )
    medium = Medium()
    trainer = CommunicationTrainer(agent1, agent2, medium)
    
    # Load puzzles
    puzzles = load_arc_puzzles('arc-agi_test_challenges.json')  
    
    # 1. Analyze puzzle symbol distribution
    symbol_dist = analyze_puzzle_distribution(puzzles)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Symbol distribution
    plt.subplot(121)
    plt.bar(symbol_dist.keys(), symbol_dist.values())
    plt.title('Symbol Distribution in Puzzles')
    plt.xlabel('Symbol Value')
    plt.ylabel('Count')
    
    # 2. Train agents briefly
    print("Training agents...")
    trainer.train(
        puzzles=puzzles[:10],  # Use subset for quick training
        num_cycles=2,
        num_exchanges=5,
        embedding_threshold=0.95
    )
    
    # 3. Analyze decoder outputs
    print("Analyzing decoder outputs...")
    first_output, differences = analyze_decoder_outputs(agent2, puzzles)
    
    # Plot 2: Decoder output analysis
    plt.subplot(122)
    plt.hist(differences, bins=20)
    plt.title('Differences in Decoder Outputs\n(compared to first output)')
    plt.xlabel('Number of Different Cells')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    
    # Print summary statistics
    print("\nAnalysis Results:")
    print(f"Most common symbol: {max(symbol_dist.items(), key=lambda x: x[1])[0]}")
    print(f"Average differences between outputs: {np.mean(differences):.2f} cells")
    print(f"Max difference between outputs: {max(differences)} cells")
    
    # If all differences are 0, decoder is producing constant output
    if np.all(np.array(differences) == 0):
        print("\nDECODER IS PRODUCING CONSTANT OUTPUT!")
        print("Constant output symbol distribution:")
        unique, counts = np.unique(first_output, return_counts=True)
        for symbol, count in zip(unique, counts):
            print(f"Symbol {symbol}: {count} cells ({count/first_output.size*100:.1f}%)")

if __name__ == "__main__":
    run_analysis()