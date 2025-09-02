import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent import Agent
from trainer import CommunicationTrainer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def track_grad_flow(named_parameters) -> Dict[str, float]:
    """Returns dictionary of gradient norms by layer"""
    grads = {}
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            grads[n] = p.grad.norm().item()
    return grads

def diagnose_gradient_flow():
    print("Initializing test...")
    torch.manual_seed(42)
    
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
    
    # Track gradient evolution over multiple steps
    grad_history = {
        'encoder': [],
        'decoder': [],
        'embedding': []
    }
    
    # Track loss evolution
    losses = []
    
    def hook_fn(grad):
        """Gradient hook for tracking max/min gradients"""
        print(f"Gradient stats: max={grad.max().item():.6f}, min={grad.min().item():.6f}, mean={grad.mean().item():.6f}")
        return grad
        
    # Register hooks for key layers
    agent1.encoder.symbol_head.weight.register_hook(lambda grad: hook_fn(grad))
    agent2.decoder.output.weight.register_hook(lambda grad: hook_fn(grad))
    
    # Create test puzzle that exercises different aspects of the model
    puzzle = torch.zeros((1, 5, 5), device=device, dtype=torch.long)
    # Add some patterns
    puzzle[0, 1:4, 1:4] = 1  # square in middle
    puzzle[0, 2, 2] = 2  # center point
    
    for step in range(50):  # Run for 50 steps
        print(f"\nStep {step + 1}")
        
        # Forward pass with intermediate value tracking
        with torch.no_grad():
            puzzle_emb = agent1.embedding_system.embed_puzzle(puzzle)
            print(f"Puzzle embedding stats: mean={puzzle_emb.mean().item():.6f}, std={puzzle_emb.std().item():.6f}")
        
        # Agent 1 encodes
        symbols1, symbol_logits1, _ = agent1.encode_puzzle_to_message(
            puzzle, temperature=1.0, initial_phase=False
        )
        
        print(f"Symbol distribution:")
        with torch.no_grad():
            symbol_probs = F.softmax(symbol_logits1, dim=-1)
            print(f"  Mean prob: {symbol_probs.mean().item():.6f}")
            print(f"  Max prob: {symbol_probs.max().item():.6f}")
            print(f"  Entropy: {-(symbol_probs * torch.log(symbol_probs + 1e-10)).sum(-1).mean().item():.6f}")
        
        # Agent 2 decodes
        reconstructed1, grid_logits1 = agent2.decode_message_to_puzzle(
            symbols1,
            target_size=(puzzle.size(1), puzzle.size(2))
        )
        
        # Compute and track loss
        grid_loss = F.cross_entropy(
            grid_logits1.reshape(-1, agent2.puzzle_symbols),
            puzzle.reshape(-1)
        )
        losses.append(grid_loss.item())
        
        # Track gradients before backward
        pre_grads = {
            'encoder': track_grad_flow(agent1.encoder.named_parameters()),
            'decoder': track_grad_flow(agent2.decoder.named_parameters()),
            'embedding': track_grad_flow(agent1.embedding_system.named_parameters())
        }
        
        # Backward pass
        grid_loss.backward()
        
        # Track gradients after backward
        grad_history['encoder'].append(track_grad_flow(agent1.encoder.named_parameters()))
        grad_history['decoder'].append(track_grad_flow(agent2.decoder.named_parameters()))
        grad_history['embedding'].append(track_grad_flow(agent1.embedding_system.named_parameters()))
        
        # Zero gradients for next step
        agent1.zero_grad()
        agent2.zero_grad()
        
        if step % 10 == 0:
            print("\nDetailed gradient analysis:")
            for name, param in agent1.encoder.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    print(f"{name}:")
                    print(f"  grad_mean: {grad.mean().item():.6f}")
                    print(f"  grad_std: {grad.std().item():.6f}")
                    print(f"  grad_norm: {grad.norm().item():.6f}")
                    print(f"  param_mean: {param.mean().item():.6f}")
                    print(f"  param_std: {param.std().item():.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss evolution
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss Evolution')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    # Plot 2: Gradient norms
    plt.subplot(2, 1, 2)
    for component in ['encoder', 'decoder', 'embedding']:
        # Take first layer as representative
        first_layer = list(grad_history[component][0].keys())[0]
        values = [step[first_layer] for step in grad_history[component]]
        plt.plot(values, label=f'{component} ({first_layer})')
    plt.title('Gradient Norm Evolution')
    plt.xlabel('Step')
    plt.ylabel('Gradient Norm')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gradient_diagnosis.png')
    plt.close()

if __name__ == "__main__":
    diagnose_gradient_flow()