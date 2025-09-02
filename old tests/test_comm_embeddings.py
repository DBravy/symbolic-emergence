import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent import Agent
from trainer import CommunicationTrainer

def debug_tensor(name, tensor, include_data=False):
    """Enhanced debug helper for tensors"""
    print(f"\n=== {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    if hasattr(tensor, 'grad_fn') and tensor.grad_fn is not None:
        print(f"Grad fn: {tensor.grad_fn}")
    
    # Safely check gradient information
    if tensor.is_leaf and tensor.requires_grad:
        if tensor.grad is not None:
            print(f"Gradient shape: {tensor.grad.shape}")
            print(f"Gradient mean: {tensor.grad.abs().mean().item():.6f}")
            print(f"Gradient max: {tensor.grad.abs().max().item():.6f}")
        else:
            print("No gradients computed yet")
    
    if include_data and tensor.numel() < 50:
        print("Data:")
        print(tensor.detach().cpu().numpy())
    print("-" * 50)

def test_bidirectional_communication():
    print("\nInitializing Bidirectional Communication Test...")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    embedding_dim = 64
    hidden_dim = 128
    num_symbols = 15
    puzzle_symbols = 10
    max_seq_length = 5
    max_grid_size = (10, 10)
    device = torch.device('cpu')
    
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
    
    # Create two different test puzzles
    puzzle1 = torch.randint(0, puzzle_symbols, (1, 5, 5), device=device)
    puzzle2 = torch.randint(0, puzzle_symbols, (1, 5, 5), device=device)
    
    print("\nPhase 1: Agent 1 -> Agent 2 Communication")
    # Agent 1 encodes puzzle1
    symbols1, logits1, _ = agent1.encode_puzzle_to_message(puzzle1, temperature=1.0)
    debug_tensor("Agent 1's Message Symbols", symbols1)
    debug_tensor("Agent 1's Communication Embedding Before", agent1.communication_embedding.weight)
    
    # Agent 2 decodes
    reconstructed1, grid_logits1 = agent2.decode_message_to_puzzle(
        symbols1,
        target_size=(puzzle1.size(1), puzzle1.size(2))
    )
    
    # Compute loss for first direction
    loss1 = F.cross_entropy(
        grid_logits1.reshape(-1, agent2.puzzle_symbols),
        puzzle1.reshape(-1)
    )
    print(f"\nLoss (Agent 1 -> Agent 2): {loss1.item():.4f}")
    
    print("\nPhase 2: Agent 2 -> Agent 1 Communication")
    # Agent 2 encodes puzzle2
    symbols2, logits2, _ = agent2.encode_puzzle_to_message(puzzle2, temperature=1.0)
    debug_tensor("Agent 2's Message Symbols", symbols2)
    debug_tensor("Agent 2's Communication Embedding Before", agent2.communication_embedding.weight)
    
    # Agent 1 decodes
    reconstructed2, grid_logits2 = agent1.decode_message_to_puzzle(
        symbols2,
        target_size=(puzzle2.size(1), puzzle2.size(2))
    )
    
    # Compute loss for second direction
    loss2 = F.cross_entropy(
        grid_logits2.reshape(-1, agent1.puzzle_symbols),
        puzzle2.reshape(-1)
    )
    print(f"\nLoss (Agent 2 -> Agent 1): {loss2.item():.4f}")
    
    # Compute total loss and backward
    total_loss = loss1 + loss2
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    
    print("\nPhase 3: Backward Pass")
    # Zero gradients
    agent1.zero_grad()
    agent2.zero_grad()
    
    # Backward pass
    total_loss.backward()
    
    print("\nPhase 4: Gradient Analysis")
    # Check gradients after backward pass
    debug_tensor("Agent 1's Communication Embedding After", agent1.communication_embedding.weight)
    debug_tensor("Agent 2's Communication Embedding After", agent2.communication_embedding.weight)
    
    # Analyze communication symbol usage
    print("\nCommunication Symbol Usage Analysis")
    print("\nAgent 1's Symbol Usage:")
    symbol_usage1 = symbols1[..., puzzle_symbols:].sum(dim=(0, 1))
    for i, usage in enumerate(symbol_usage1):
        print(f"Symbol {i + puzzle_symbols}: {usage.item():.4f}")
    
    print("\nAgent 2's Symbol Usage:")
    symbol_usage2 = symbols2[..., puzzle_symbols:].sum(dim=(0, 1))
    for i, usage in enumerate(symbol_usage2):
        print(f"Symbol {i + puzzle_symbols}: {usage.item():.4f}")
    
    # Check if gradients are flowing through the projected space
    print("\nGradient Flow Analysis")
    comm_grads1 = agent1.communication_embedding.weight.grad
    comm_grads2 = agent2.communication_embedding.weight.grad
    
    if comm_grads1 is not None:
        print("\nAgent 1 Gradient Stats:")
        print(f"Mean gradient magnitude: {comm_grads1.abs().mean().item():.6f}")
        print(f"Max gradient magnitude: {comm_grads1.abs().max().item():.6f}")
    
    if comm_grads2 is not None:
        print("\nAgent 2 Gradient Stats:")
        print(f"Mean gradient magnitude: {comm_grads2.abs().mean().item():.6f}")
        print(f"Max gradient magnitude: {comm_grads2.abs().max().item():.6f}")

if __name__ == "__main__":
    test_bidirectional_communication()