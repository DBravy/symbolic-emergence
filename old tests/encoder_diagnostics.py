import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

# Import required modules from your codebase
from agent import Agent
from puzzle import Puzzle
from embeddings import PuzzleEmbedding

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create simple test puzzles
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

def test_encoder_gradient_flow(encoder, puzzle_tensor):
    """Test gradient flow through the encoder"""
    # Reset gradients
    for param in encoder.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # Forward pass
    puzzle_emb = encoder.embedding_system.embed_puzzle(puzzle_tensor)
    comm_embeddings = encoder.communication_embedding.weight[encoder.puzzle_symbols:]
    length_logits, symbol_logits = encoder.encoder(puzzle_emb, comm_embeddings)
    
    # Create a simple loss (just maximize the first symbol's probability)
    target_symbol = 0
    loss = -symbol_logits[0, 0, target_symbol]
    
    # Backward pass
    loss.backward()
    
    # Collect gradient statistics
    grad_stats = {}
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                'mean': param.grad.abs().mean().item(),
                'std': param.grad.std().item(),
                'min': param.grad.min().item(),
                'max': param.grad.max().item()
            }
    
    return grad_stats

def test_encoder_consistency(encoder, puzzle_tensors, num_trials=10):
    """Test if the encoder produces consistent outputs for the same input"""
    consistency_results = []
    
    for puzzle_idx, puzzle_tensor in enumerate(puzzle_tensors):
        symbol_counts = defaultdict(int)
        
        for _ in range(num_trials):
            # Forward pass with temperature 0.1 (low randomness)
            symbols, _, _ = encoder.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
            
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

def test_simple_classification(encoder, puzzle_tensors, epochs=100):
    """
    Test if the encoder can learn a simple classification task:
    Map each distinct puzzle to a specific target symbol
    """
    # Assign each puzzle a target symbol
    targets = torch.tensor([i % (encoder.num_symbols - encoder.puzzle_symbols) 
                           for i in range(len(puzzle_tensors))], 
                          device=device)
    
    # Create optimizer for encoder parameters
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    history = {
        'loss': [],
        'accuracy': []
    }
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i, puzzle_tensor in enumerate(puzzle_tensors):
            optimizer.zero_grad()
            
            # Forward pass
            puzzle_emb = encoder.embedding_system.embed_puzzle(puzzle_tensor)
            comm_embeddings = encoder.communication_embedding.weight[encoder.puzzle_symbols:]
            _, symbol_logits = encoder.encoder(puzzle_emb, comm_embeddings)
            
            # Get target symbol
            target = targets[i]
            
            # Compute loss
            loss = F.cross_entropy(symbol_logits[0, 0].unsqueeze(0), target.unsqueeze(0))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check accuracy
            pred = symbol_logits[0, 0].argmax().item()
            if pred == target.item():
                correct += 1
        
        # Record metrics
        avg_loss = total_loss / len(puzzle_tensors)
        accuracy = correct / len(puzzle_tensors)
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}")
    
    return history

def test_contrastive_learning(encoder, puzzle_tensors, epochs=100):
    """
    Test if the encoder can learn with a simplified contrastive objective:
    Make same puzzles have similar encodings, different puzzles have different encodings
    """
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    history = {
        'loss': [],
        'positive_sim': [],
        'negative_sim': []
    }
    
    for epoch in range(epochs):
        total_loss = 0
        avg_pos_sim = 0
        avg_neg_sim = 0
        
        # Process each puzzle
        for i, anchor in enumerate(puzzle_tensors):
            optimizer.zero_grad()
            
            # Create a positive pair (slightly modified version)
            positive = anchor.clone()
            mask = torch.rand_like(positive.float()) < 0.05  # 5% noise
            positive[mask] = torch.randint_like(positive[mask], 0, encoder.puzzle_symbols)
            
            # Choose a negative (different puzzle)
            neg_idx = (i + 1) % len(puzzle_tensors)  # simple cycling
            negative = puzzle_tensors[neg_idx]
            
            # Get encodings
            anchor_emb = encoder.embedding_system.embed_puzzle(anchor)
            positive_emb = encoder.embedding_system.embed_puzzle(positive)
            negative_emb = encoder.embedding_system.embed_puzzle(negative)
            
            # Flatten embeddings to get vector representations
            anchor_vec = anchor_emb.mean(dim=1)   # [B, D]
            positive_vec = positive_emb.mean(dim=1)  # [B, D]
            negative_vec = negative_emb.mean(dim=1)  # [B, D]
            
            # Normalize
            anchor_norm = F.normalize(anchor_vec, p=2, dim=1)
            positive_norm = F.normalize(positive_vec, p=2, dim=1)
            negative_norm = F.normalize(negative_vec, p=2, dim=1)
            
            # Compute similarities
            pos_sim = F.cosine_similarity(anchor_norm, positive_norm)
            neg_sim = F.cosine_similarity(anchor_norm, negative_norm)
            
            # Contrastive loss: push pos_sim to 1, neg_sim to -1
            loss = (1 - pos_sim) + torch.clamp(neg_sim - 0.5, min=0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_pos_sim += pos_sim.item()
            avg_neg_sim += neg_sim.item()
        
        # Record metrics
        avg_loss = total_loss / len(puzzle_tensors)
        avg_pos_sim /= len(puzzle_tensors)
        avg_neg_sim /= len(puzzle_tensors)
        
        history['loss'].append(avg_loss)
        history['positive_sim'].append(avg_pos_sim)
        history['negative_sim'].append(avg_neg_sim)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Pos_Sim={avg_pos_sim:.2f}, Neg_Sim={avg_neg_sim:.2f}")
    
    return history

def test_symbol_distribution(encoder, puzzle_tensors):
    """Test the distribution of symbols that the encoder assigns to puzzles"""
    symbol_distribution = []
    
    for puzzle_idx, puzzle in enumerate(puzzle_tensors):
        # Forward pass with low temperature for more deterministic output
        symbols, symbol_logits, _ = encoder.encode_puzzle_to_message(puzzle, temperature=0.1)
        
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

def plot_test_results(history, title, filename):
    """Plot test results"""
    plt.figure(figsize=(12, 6))
    
    # Handle different history formats
    if 'loss' in history and 'accuracy' in history:
        # Classification results
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'])
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
    elif 'loss' in history and 'positive_sim' in history:
        # Contrastive results
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['positive_sim'], label='Positive Similarity')
        plt.plot(history['negative_sim'], label='Negative Similarity')
        plt.title(f'{title} - Similarities')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def print_gradient_stats(grad_stats):
    """Print gradient statistics in a readable format"""
    print("\n=== Gradient Statistics ===")
    
    # Group parameters by module
    module_stats = defaultdict(list)
    for name, stats in grad_stats.items():
        # Extract module name (first part of parameter name)
        module = name.split('.')[0]
        module_stats[module].append((name, stats))
    
    for module, params in module_stats.items():
        print(f"\n== {module} ==")
        for name, stats in params:
            print(f"  {name}:")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std:  {stats['std']:.6f}")
            print(f"    Min:  {stats['min']:.6f}")
            print(f"    Max:  {stats['max']:.6f}")

def test_parameter_updates(encoder, puzzle_tensor, steps=10):
    """Test if parameters are actually being updated during training"""
    # Get initial parameter values
    initial_params = {}
    for name, param in encoder.named_parameters():
        initial_params[name] = param.data.clone()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)  # Higher LR to see clear changes
    
    # Train for a few steps
    changes = {}
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        puzzle_emb = encoder.embedding_system.embed_puzzle(puzzle_tensor)
        comm_embeddings = encoder.communication_embedding.weight[encoder.puzzle_symbols:]
        _, symbol_logits = encoder.encoder(puzzle_emb, comm_embeddings)
        
        # Simple loss (just maximize first symbol)
        loss = -symbol_logits[0, 0, 0]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Check parameter changes
        if step == steps - 1:  # Only check at the end
            for name, param in encoder.named_parameters():
                initial = initial_params[name]
                current = param.data
                # Calculate absolute change
                change = (current - initial).abs().mean().item()
                changes[name] = change
    
    return changes

def run_encoder_diagnosis():
    """Run a comprehensive diagnosis of encoder issues"""
    # Create test puzzles
    test_puzzles = create_test_puzzles(num_puzzles=5)
    
    # Create encoder agent
    encoder = Agent(
        agent_id="encoder_test",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=12,
        puzzle_symbols=10,
        max_seq_length=1
    ).to(device)
    
    # Convert puzzles to tensors
    puzzle_tensors = [
        torch.tensor(puzzle.test_input, dtype=torch.long, device=device).unsqueeze(0)
        for puzzle in test_puzzles
    ]
    
    # Test 1: Check encoder consistency before training
    print("\n==== Test 1: Encoder Consistency (Before Training) ====")
    initial_consistency = test_encoder_consistency(encoder, puzzle_tensors)
    for result in initial_consistency:
        print(f"Puzzle {result['puzzle_idx']}: Consistency {result['consistency']:.2f}")
        print(f"  Symbol counts: {result['symbol_counts']}")
    
    # Test 2: Check gradient flow
    print("\n==== Test 2: Gradient Flow Analysis ====")
    grad_stats = test_encoder_gradient_flow(encoder, puzzle_tensors[0])
    print_gradient_stats(grad_stats)
    
    # Test 3: Check if parameters actually update
    print("\n==== Test 3: Parameter Update Check ====")
    param_changes = test_parameter_updates(encoder, puzzle_tensors[0])
    print("\nParameter changes after 10 steps:")
    
    # Group by module for readability
    module_changes = defaultdict(list)
    for name, change in param_changes.items():
        module = name.split('.')[0]
        module_changes[module].append((name, change))
    
    for module, changes in module_changes.items():
        print(f"\n== {module} ==")
        for name, change in changes:
            if change > 0:
                print(f"  {name}: {change:.6f}")
            else:
                print(f"  {name}: {change:.6f} (NO CHANGE)")
    
    # Test 4: Simple classification test
    print("\n==== Test 4: Simple Classification Training ====")
    classification_history = test_simple_classification(encoder, puzzle_tensors, epochs=100)
    plot_test_results(classification_history, 
                      "Simple Classification", 
                      "classification_results.png")
    
    # Test 5: Contrastive learning test
    print("\n==== Test 5: Simplified Contrastive Learning ====")
    contrastive_history = test_contrastive_learning(encoder, puzzle_tensors, epochs=100)
    plot_test_results(contrastive_history, 
                     "Simplified Contrastive Learning", 
                     "contrastive_results.png")
    
    # Test 6: Check encoder consistency after training
    print("\n==== Test 6: Encoder Consistency (After Training) ====")
    final_consistency = test_encoder_consistency(encoder, puzzle_tensors)
    for result in final_consistency:
        print(f"Puzzle {result['puzzle_idx']}: Consistency {result['consistency']:.2f}")
        print(f"  Symbol counts: {result['symbol_counts']}")
    
    # Test 7: Check symbol distribution
    print("\n==== Test 7: Symbol Distribution Analysis ====")
    symbol_dist = test_symbol_distribution(encoder, puzzle_tensors)
    
    # Count how many unique symbols are used
    used_symbols = set()
    for dist in symbol_dist:
        used_symbols.add(dist['assigned_symbol'])
        
        print(f"Puzzle {dist['puzzle_idx']}: Assigned symbol {dist['assigned_symbol']}")
        # Print top 3 probabilities
        probs = dist['symbol_probs']
        top_indices = np.argsort(-probs)[:3]
        for idx in top_indices:
            print(f"  Symbol {idx}: {probs[idx]:.4f}")
    
    print(f"\nUnique symbols used: {len(used_symbols)} out of {len(symbol_dist)}")
    
    print("\n==== Diagnosis Summary ====")
    print(f"1. Initial consistency: {np.mean([r['consistency'] for r in initial_consistency]):.2f}")
    print(f"2. Final consistency: {np.mean([r['consistency'] for r in final_consistency]):.2f}")
    print(f"3. Unique symbols used: {len(used_symbols)} out of {len(symbol_dist)}")
    print(f"4. Classification final accuracy: {classification_history['accuracy'][-1]:.2f}")
    print(f"5. Contrastive final loss: {contrastive_history['loss'][-1]:.4f}")
    
    # Provide diagnosis based on results
    print("\n==== Potential Issues ====")
    
    # Check gradient flow
    low_gradients = any(stats['mean'] < 1e-6 for stats in grad_stats.values())
    if low_gradients:
        print("✘ Gradient flow problem: Some components have very small gradients")
    else:
        print("✓ Gradient flow seems acceptable")
    
    # Check parameter updates
    no_updates = any(change < 1e-6 for change in param_changes.values())
    if no_updates:
        print("✘ Some parameters are not being updated during training")
    else:
        print("✓ Parameters are being updated properly")
    
    # Check classification learning
    if classification_history['accuracy'][-1] < 0.8:
        print("✘ Encoder struggles with a simple classification task")
    else:
        print("✓ Encoder can learn simple classification")
    
    # Check contrastive learning
    pos_sim = contrastive_history['positive_sim'][-1]
    neg_sim = contrastive_history['negative_sim'][-1]
    if pos_sim - neg_sim < 0.3:
        print("✘ Encoder struggles with contrastive learning")
    else:
        print("✓ Encoder can learn from contrastive signals")
    
    # Check consistency improvement
    initial_avg = np.mean([r['consistency'] for r in initial_consistency])
    final_avg = np.mean([r['consistency'] for r in final_consistency])
    if final_avg - initial_avg < 0.2:
        print("✘ Encoder consistency doesn't improve significantly after training")
    else:
        print("✓ Encoder consistency improves after training")
    
    # Check symbol differentiation
    if len(used_symbols) < len(puzzle_tensors) * 0.8:
        print("✘ Encoder doesn't assign different symbols to different puzzles")
    else:
        print("✓ Encoder differentiates puzzles with different symbols")

if __name__ == "__main__":
    run_encoder_diagnosis()