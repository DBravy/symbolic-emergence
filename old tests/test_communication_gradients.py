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

def analyze_symbol_probabilities(logits, puzzle_symbols):
    """Analyze symbol probability distributions"""
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        comm_probs = probs[..., puzzle_symbols:]
        
        print(f"\nSymbol Probability Analysis:")
        print(f"Communication symbols mean prob: {comm_probs.mean().item():.4f}")
        print(f"Communication symbols max prob: {comm_probs.max().item():.4f}")
        
        # Analyze entropy
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1).mean()
        print(f"Symbol distribution entropy: {entropy.item():.4f}")

def test_communication_gradients():
    print("\nInitializing Enhanced Communication Gradient Test...")
    
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
    
    # Enable gradient tracking for intermediate values
    torch.set_grad_enabled(True)
    
    # Create test puzzle
    puzzle = torch.randint(0, puzzle_symbols, (1, 5, 5), device=device)
    
    print("\nStep 1: Puzzle Embedding")
    puzzle_emb = agent1.embedding_system.embed_puzzle(puzzle)
    debug_tensor("Puzzle Embedding", puzzle_emb)
    
    print("\nStep 2: Encoding Process")
    # Get encoder outputs
    length_logits, symbol_logits = agent1.encoder(puzzle_emb)
    debug_tensor("Symbol Logits", symbol_logits)
    analyze_symbol_probabilities(symbol_logits, puzzle_symbols)
    
    print("\nStep 3: Message Generation")
    symbols1, symbol_logits1, _ = agent1.encode_puzzle_to_message(
        puzzle, temperature=1.0, initial_phase=False
    )
    debug_tensor("Generated Symbols", symbols1)
    
    print("\nStep 4: Message Embedding")
    # Explicitly compute embedded message
    embedded_message = torch.einsum('bsv,ve->bse', symbols1, agent2.communication_embedding.weight)
    debug_tensor("Embedded Message", embedded_message)
    
    print("\nStep 5: Decoding Process")
    reconstructed1, grid_logits1 = agent2.decode_message_to_puzzle(
        symbols1,
        target_size=(puzzle.size(1), puzzle.size(2)),
        temperature=1.0
    )
    debug_tensor("Grid Logits", grid_logits1)
    
    print("\nStep 6: Computing Loss")
    grid_loss = F.cross_entropy(
        grid_logits1.reshape(-1, agent2.puzzle_symbols),
        puzzle.reshape(-1)
    )
    print(f"Grid Loss: {grid_loss.item():.4f}")
    
    print("\nStep 7: Backward Pass")
    # Zero gradients
    agent1.zero_grad()
    agent2.zero_grad()
    
    # Backward pass
    grid_loss.backward()
    
    print("\nStep 8: Gradient Analysis")
    
    # Analyze Agent 1's components
    print("\nAgent 1 Gradient Analysis:")
    debug_tensor(
        "Embedding System (Comm Symbols)", 
        agent1.embedding_system.symbol_embedding.weight[puzzle_symbols:],
        include_data=True
    )
    debug_tensor(
        "Communication Embedding",
        agent1.communication_embedding.weight,
        include_data=True
    )
    
    # Analyze Agent 2's components
    print("\nAgent 2 Gradient Analysis:")
    debug_tensor(
        "Embedding System (Comm Symbols)",
        agent2.embedding_system.symbol_embedding.weight[puzzle_symbols:],
        include_data=True
    )
    debug_tensor(
        "Communication Embedding",
        agent2.communication_embedding.weight,
        include_data=True
    )
    
    print("\nStep 9: Symbol Usage Analysis")
    with torch.no_grad():
        symbol_usage = symbols1[..., puzzle_symbols:].sum(dim=(0, 1))
        print("\nCommunication Symbol Usage:")
        for i, usage in enumerate(symbol_usage):
            print(f"Symbol {i + puzzle_symbols}: {usage.item():.4f}")
        
        # Analyze which puzzle symbols were used in input
        puzzle_usage = torch.bincount(puzzle.flatten(), minlength=puzzle_symbols)
        print("\nPuzzle Symbol Usage in Input:")
        for i, count in enumerate(puzzle_usage):
            print(f"Symbol {i}: {count.item()}")

if __name__ == "__main__":
    test_communication_gradients()