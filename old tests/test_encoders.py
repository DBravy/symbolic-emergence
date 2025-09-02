"""
Test script to evaluate different encoder variants to solve the 50/50 probability issue.
This script loads each variant and performs targeted tests to see if they overcome the problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import sys
import os

# Import the encoder variants
from encoder_variants import (
    DirectClassificationEncoder,
    UnnormalizedSimilarityEncoder,
    EnhancedSimilarityEncoder,
    HybridEncoder
)

# Import base components needed for testing
from agent import Agent
from puzzle import Puzzle
from embeddings import PuzzleEmbedding

# Create a custom build_encoder function that will be used to patch the Agent class
def create_build_encoder_function(encoder_variant):
    def custom_build_encoder(embedding_dim, hidden_dim, num_symbols, puzzle_symbols, max_seq_length):
        if encoder_variant == "direct":
            return DirectClassificationEncoder(
                embedding_dim=embedding_dim,
                num_symbols=num_symbols,
                puzzle_symbols=puzzle_symbols,
                max_seq_length=max_seq_length
            )
        elif encoder_variant == "unnormalized":
            return UnnormalizedSimilarityEncoder(
                embedding_dim=embedding_dim,
                num_symbols=num_symbols,
                puzzle_symbols=puzzle_symbols,
                max_seq_length=max_seq_length
            )
        elif encoder_variant == "enhanced":
            return EnhancedSimilarityEncoder(
                embedding_dim=embedding_dim,
                num_symbols=num_symbols,
                puzzle_symbols=puzzle_symbols,
                max_seq_length=max_seq_length
            )
        elif encoder_variant == "hybrid":
            return HybridEncoder(
                embedding_dim=embedding_dim,
                num_symbols=num_symbols,
                puzzle_symbols=puzzle_symbols,
                max_seq_length=max_seq_length
            )
        else:
            raise ValueError(f"Unknown encoder variant: {encoder_variant}")
    return custom_build_encoder

def create_test_puzzles(num_puzzles=5, size=(5, 5)):
    """Create simple test puzzles with clearly distinct patterns"""
    test_puzzles = []
    
    # Create basic patterns
    patterns = [
        # Horizontal lines
        lambda i, j: 1 if i == j else 0,
        # Vertical lines
        lambda i, j: 1 if i == 0 else 0,
        # Diagonal
        lambda i, j: 1 if i == size[0]//2 else 0,
        # Checkerboard
        lambda i, j: 1 if (i + j) % 2 == 0 else 0,
        # Border
        lambda i, j: 1 if i == 0 or i == size[0]-1 or j == 0 or j == size[1]-1 else 0
    ]
    
    for p in range(min(num_puzzles, len(patterns))):
        # Create a grid with the pattern
        grid = np.zeros(size, dtype=np.int32)
        for i in range(size[0]):
            for j in range(size[1]):
                grid[i, j] = patterns[p](i, j)
        
        # Create a puzzle with the pattern
        puzzle = Puzzle.from_single_example(grid, grid)
        test_puzzles.append(puzzle)
    
    # If more puzzles needed, add variations
    if num_puzzles > len(patterns):
        for p in range(len(patterns), num_puzzles):
            pattern_idx = p % len(patterns)
            # Create a variation by adding some noise
            grid = np.zeros(size, dtype=np.int32)
            for i in range(size[0]):
                for j in range(size[1]):
                    grid[i, j] = patterns[pattern_idx](i, j)
            
            # Add some noise
            noise = np.random.randint(0, 2, size=grid.shape)
            grid = (grid + noise) % 2
            
            puzzle = Puzzle.from_single_example(grid, grid)
            test_puzzles.append(puzzle)
    
    return test_puzzles

def test_encoder_consistency(agent, puzzle_tensors, num_trials=10):
    """Test if the encoder produces consistent outputs for the same input"""
    from collections import defaultdict
    consistency_results = []
    
    for puzzle_idx, puzzle_tensor in enumerate(puzzle_tensors):
        symbol_counts = defaultdict(int)
        
        for _ in range(num_trials):
            # Forward pass with temperature 0.1 (low randomness)
            symbols, _, _ = agent.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
            
            # Get the argmax symbol
            symbol = symbols[0, 0].argmax().item()
            symbol_counts[symbol] += 1
        
        # Calculate consistency as percentage of most common symbol
        most_common = max(symbol_counts.values())
        consistency = most_common / num_trials
        
        consistency_results.append({
            'puzzle_idx': puzzle_idx,
            'consistency': consistency,
            'symbol_counts': dict(symbol_counts)
        })
    
    return consistency_results

def test_symbol_distribution(agent, puzzle_tensors):
    """Test the distribution of symbols that the encoder assigns to puzzles"""
    symbol_distribution = []
    
    for puzzle_idx, puzzle in enumerate(puzzle_tensors):
        # Forward pass with low temperature for more deterministic output
        symbols, symbol_logits, _ = agent.encode_puzzle_to_message(puzzle, temperature=0.1)
        
        # Get the argmax symbol
        symbol = symbols[0, 0].argmax().item()
        
        # Get distribution over symbols
        probs = F.softmax(symbol_logits[0, 0], dim=-1).detach().cpu().numpy()
        
        symbol_distribution.append({
            'puzzle_idx': puzzle_idx,
            'assigned_symbol': symbol,
            'symbol_probs': probs
        })
    
    return symbol_distribution

def train_encoder(agent, puzzle_tensors, num_epochs=100):
    """Simple training loop for the encoder"""
    # Create optimizer for encoder parameters
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    device = next(agent.parameters()).device
    
    # Assign different target symbols to different puzzles
    targets = {}
    for i in range(len(puzzle_tensors)):
        targets[i] = i % (agent.num_symbols - agent.puzzle_symbols)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        
        for i, puzzle in enumerate(puzzle_tensors):
            optimizer.zero_grad()
            
            # Forward pass
            symbols, symbol_logits, _ = agent.encode_puzzle_to_message(puzzle, temperature=0.1)
            
            # Create proper target tensor
            target = torch.tensor([targets[i]], device=device)
            
            # Use cross-entropy loss with the full probability distribution
            loss = F.cross_entropy(symbol_logits[0, 0].unsqueeze(0), target)
            
            # Add L2 regularization to prevent extreme values
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in agent.encoder.parameters())
            loss = loss + l2_reg
            
            # Add diversity loss to encourage different symbols for different puzzles
            if len(puzzle_tensors) > 1:
                all_preds = []
                for j, other_puzzle in enumerate(puzzle_tensors):
                    if j != i:  # Only compare to other puzzles
                        with torch.no_grad():
                            _, other_logits, _ = agent.encode_puzzle_to_message(other_puzzle, temperature=0.1)
                            other_probs = F.softmax(other_logits[0, 0], dim=-1)
                            all_preds.append(other_probs)
                
                # Current puzzle probabilities
                current_probs = F.softmax(symbol_logits[0, 0], dim=-1)
                
                # Penalize similarity to other puzzles
                diversity_loss = 0.0
                for other_probs in all_preds:
                    similarity = F.cosine_similarity(current_probs.unsqueeze(0), other_probs.unsqueeze(0))
                    diversity_loss += similarity
                
                if len(all_preds) > 0:
                    loss = loss + 0.1 * (diversity_loss / len(all_preds))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Tracking
            pred_symbol = symbols[0, 0].argmax().item()
            if pred_symbol == targets[i]:
                correct += 1
            total_loss += loss.item()
        
        # Record metrics
        avg_loss = total_loss / len(puzzle_tensors)
        accuracy = correct / len(puzzle_tensors)
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}")
    
    return history

def plot_distribution(variant_name, distribution):
    """Plot symbol probability distributions for each puzzle"""
    num_puzzles = len(distribution)
    num_symbols = len(distribution[0]['symbol_probs'])
    
    plt.figure(figsize=(12, 8))
    
    for i, dist in enumerate(distribution):
        plt.subplot(num_puzzles, 1, i+1)
        probs = dist['symbol_probs']
        plt.bar(range(num_symbols), probs)
        plt.title(f"Puzzle {i} (Assigned Symbol: {dist['assigned_symbol']})")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        # Add probability values above bars
        for j, v in enumerate(probs):
            if v > 0.05:  # Only show values above 5%
                plt.text(j, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.xlabel("Symbol")
    plt.tight_layout()
    plt.savefig(f"{variant_name}_distribution.png")
    plt.close()

def test_variant(variant_name, device):
    """Test a specific encoder variant"""
    print(f"\n===== Testing {variant_name.upper()} Encoder Variant =====")
    
    # Create agent with the specified encoder variant
    custom_build_encoder = create_build_encoder_function(variant_name)
    
    # Create a small agent for testing
    agent = Agent(
        agent_id=f"{variant_name}_test",
        embedding_dim=256,  # Smaller for faster testing
        hidden_dim=512,
        num_symbols=12,
        puzzle_symbols=10,
        max_seq_length=1,
        encoder=custom_build_encoder(256, 512, 12, 10, 1)
    ).to(device)
    
    # Create test puzzles
    test_puzzles = create_test_puzzles(num_puzzles=5)
    
    # Convert puzzles to tensors
    puzzle_tensors = [
        torch.tensor(puzzle.test_input, dtype=torch.long, device=device).unsqueeze(0)
        for puzzle in test_puzzles
    ]
    
    # Test 1: Initial consistency
    print("\n1. Initial Encoder Consistency (Before Training)")
    initial_consistency = test_encoder_consistency(agent, puzzle_tensors)
    for result in initial_consistency:
        print(f"Puzzle {result['puzzle_idx']}: Consistency {result['consistency']:.2f}")
        print(f"  Symbol counts: {result['symbol_counts']}")
    
    # Test 2: Initial symbol distribution
    print("\n2. Initial Symbol Distribution")
    initial_distribution = test_symbol_distribution(agent, puzzle_tensors)
    for dist in initial_distribution:
        print(f"Puzzle {dist['puzzle_idx']}: Assigned symbol {dist['assigned_symbol']}")
        # Print top 3 probabilities
        probs = dist['symbol_probs']
        for j, p in enumerate(probs):
            print(f"  Symbol {j}: {p:.4f}")
    
    # Count initial unique symbols
    initial_symbols = set(dist['assigned_symbol'] for dist in initial_distribution)
    print(f"\nInitial unique symbols used: {len(initial_symbols)} out of {len(initial_distribution)}")
    
    # Plot initial distributions
    plot_distribution(f"{variant_name}_initial", initial_distribution)
    
    # Test 3: Train the encoder
    print("\n3. Training Encoder")
    history = train_encoder(agent, puzzle_tensors, num_epochs=100)
    
    # Test 4: Check consistency after training
    print("\n4. Final Encoder Consistency (After Training)")
    final_consistency = test_encoder_consistency(agent, puzzle_tensors)
    for result in final_consistency:
        print(f"Puzzle {result['puzzle_idx']}: Consistency {result['consistency']:.2f}")
        print(f"  Symbol counts: {result['symbol_counts']}")
    
    # Test 5: Final symbol distribution
    print("\n5. Final Symbol Distribution")
    final_distribution = test_symbol_distribution(agent, puzzle_tensors)
    for dist in final_distribution:
        print(f"Puzzle {dist['puzzle_idx']}: Assigned symbol {dist['assigned_symbol']}")
        # Print top probabilities
        probs = dist['symbol_probs']
        for j, p in enumerate(probs):
            print(f"  Symbol {j}: {p:.4f}")
    
    # Count final unique symbols
    final_symbols = set(dist['assigned_symbol'] for dist in final_distribution)
    print(f"\nFinal unique symbols used: {len(final_symbols)} out of {len(final_distribution)}")
    
    # Plot final distributions
    plot_distribution(f"{variant_name}_final", final_distribution)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Symbol Assignment Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{variant_name}_training.png")
    plt.close()
    
    # Summary
    print("\n===== Variant Summary =====")
    print(f"1. Initial consistency: {np.mean([r['consistency'] for r in initial_consistency]):.2f}")
    print(f"2. Final consistency: {np.mean([r['consistency'] for r in final_consistency]):.2f}")
    print(f"3. Initial unique symbols: {len(initial_symbols)} out of {len(initial_distribution)}")
    print(f"4. Final unique symbols: {len(final_symbols)} out of {len(final_distribution)}")
    print(f"5. Final training accuracy: {history['accuracy'][-1]:.2f}")
    
    success = len(final_symbols) > len(initial_symbols) and history['accuracy'][-1] > 0.7
    print(f"OVERALL ASSESSMENT: {'SUCCESS' if success else 'FAILURE'}")
    
    return {
        'variant': variant_name,
        'initial_consistency': np.mean([r['consistency'] for r in initial_consistency]),
        'final_consistency': np.mean([r['consistency'] for r in final_consistency]),
        'initial_unique_symbols': len(initial_symbols),
        'final_unique_symbols': len(final_symbols),
        'final_accuracy': history['accuracy'][-1],
        'success': success
    }

def run_all_tests():
    """Run tests on all encoder variants"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    variants = ["direct", "unnormalized", "enhanced", "hybrid"]
    results = []
    
    for variant in variants:
        result = test_variant(variant, device)
        results.append(result)
    
    # Print comparison table
    print("\n===== Variant Comparison =====")
    print(f"{'Variant':<15} {'Initial Cons.':<15} {'Final Cons.':<15} {'Initial Sym.':<15} {'Final Sym.':<15} {'Accuracy':<15} {'Result':<10}")
    print("-" * 90)
    
    for r in results:
        success_str = "SUCCESS" if r['success'] else "FAILURE"
        print(f"{r['variant']:<15} {r['initial_consistency']:<15.2f} {r['final_consistency']:<15.2f} {r['initial_unique_symbols']:<15} {r['final_unique_symbols']:<15} {r['final_accuracy']:<15.2f} {success_str:<10}")
    
    # Find best variant
    best_variant = max(results, key=lambda x: x['final_accuracy'])
    print(f"\nBest variant: {best_variant['variant'].upper()} (Accuracy: {best_variant['final_accuracy']:.2f})")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    metrics = [
        ('initial_consistency', 'Initial Consistency', 'Consistency'),
        ('final_consistency', 'Final Consistency', 'Consistency'),
        ('initial_unique_symbols', 'Initial Unique Symbols', 'Count'),
        ('final_unique_symbols', 'Final Unique Symbols', 'Count'),
        ('final_accuracy', 'Final Accuracy', 'Accuracy')
    ]
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        values = [r[metric] for r in results]
        plt.bar(range(len(variants)), values)
        plt.xticks(range(len(variants)), [v.capitalize() for v in variants], rotation=45)
        plt.title(title)
        plt.ylabel(ylabel)
        
        # Add values above bars
        for j, v in enumerate(values):
            plt.text(j, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("encoder_comparison.png")
    plt.close()
    
    print("\nComparison plots saved to encoder_comparison.png")
    for variant in variants:
        print(f"- {variant}_initial_distribution.png")
        print(f"- {variant}_final_distribution.png")
        print(f"- {variant}_training.png")

if __name__ == "__main__":
    run_all_tests()