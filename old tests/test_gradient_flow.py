import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent import Agent
from trainer import CommunicationTrainer

def print_gradient_flow(named_parameters):
    """Prints the gradient flow for model parameters."""
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            print(f"Layer: {n}")
            if p.grad is not None:
                print(f"  Average gradient: {p.grad.abs().mean().item():.10f}")
                print(f"  Max gradient: {p.grad.abs().max().item():.10f}")
                print(f"  Shape: {p.grad.shape}")
            else:
                print("  NO GRADIENT")
            print(f"  Requires grad: {p.requires_grad}")
            print("-" * 50)

def debug_tensor(name, tensor):
    """Debug helper for tensors"""
    print(f"\nDebug {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Has grad fn: {tensor.grad_fn is not None}")
    if tensor.grad_fn is not None:
        print(f"  Grad fn: {tensor.grad_fn}")
    print(f"  Is leaf: {tensor.is_leaf}")
    print("-" * 50)

def test_gradient_flow():
    print("Initializing test...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cpu')
    
    # Configuration
    embedding_dim = 64
    hidden_dim = 128
    num_symbols = 15
    puzzle_symbols = 10
    max_seq_length = 5
    max_grid_size = (10, 10)
    
    # Create agents
    agent1 = Agent(
        agent_id='agent1',
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        max_grid_size=max_grid_size
    ).to(device)
    
    agent2 = Agent(
        agent_id='agent2',
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        max_grid_size=max_grid_size
    ).to(device)
    
    # Create trainer
    trainer = CommunicationTrainer(agent1=agent1, agent2=agent2, learning_rate=0.001)
    
    # Create test puzzle
    puzzle = torch.randint(0, puzzle_symbols, (1, 5, 5), device=device)
    
    print("\nStarting encoder debug...")
    
    # Debug embedding process
    puzzle_emb = agent1.embedding_system.embed_puzzle(puzzle)
    debug_tensor("puzzle_embedding", puzzle_emb)
    
    # Debug encoder outputs
    length_logits, symbol_logits = agent1.encoder(puzzle_emb)
    debug_tensor("length_logits", length_logits)
    debug_tensor("symbol_logits", symbol_logits)
    
    # Debug message creation
    symbols1, symbol_logits1, length_stats1 = agent1.encode_puzzle_to_message(
        puzzle, temperature=1.0, initial_phase=False
    )
    debug_tensor("symbols1", symbols1)
    debug_tensor("symbol_logits1", symbol_logits1)
    
    # Forward pass through decoder
    reconstructed1, grid_logits1 = agent2.decode_message_to_puzzle(
        symbols1,
        target_size=(puzzle.size(1), puzzle.size(2))
    )
    
    print("\nComputing loss...")
    # Compute loss
    grid_loss1 = F.cross_entropy(
        grid_logits1.reshape(-1, agent2.puzzle_symbols),
        puzzle.reshape(-1)
    )
    
    print(f"Loss value: {grid_loss1.item()}")
    print(f"Loss requires grad: {grid_loss1.requires_grad}")
    print(f"Loss grad_fn: {grid_loss1.grad_fn}")
    
    print("\nStarting backward pass...")
    grid_loss1.backward()
    
    print("\nGradient Analysis for Embedding System:")
    print("=" * 50)
    print_gradient_flow(agent1.embedding_system.named_parameters())
    
    print("\nGradient Analysis for Encoder:")
    print("=" * 50)
    print_gradient_flow(agent1.encoder.named_parameters())
    
    print("\nGradient Analysis for Communication Embedding:")
    print("=" * 50)
    print_gradient_flow([('communication_embedding', agent1.communication_embedding.weight)])
    
    # Check if symbols tensor kept its gradient
    print("\nSymbols tensor after backward:")
    debug_tensor("symbols1 (after backward)", symbols1)

if __name__ == "__main__":
    test_gradient_flow()