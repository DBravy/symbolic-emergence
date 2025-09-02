import torch
from agent import Agent
from trainer import CommunicationTrainer
import numpy as np
from puzzle import Puzzle

def check_gradient_paths(agent1, agent2, puzzle_tensor):
    """Verify gradient computation paths are connected"""
    def requires_grad_info(tensor, name):
        if isinstance(tensor, tuple):
            return {f"{name}_{i}": t.requires_grad for i, t in enumerate(tensor)}
        return {name: tensor.requires_grad}

    trainer = CommunicationTrainer(agent1, agent2)
    
    # Forward pass with gradient tracking
    symbols1, symbol_logits1, _ = agent1.encode_puzzle_to_message(puzzle_tensor)
    reconstructed1, grid_logits1 = agent2.decode_message_to_puzzle(symbols1)
    
    # Check intermediate values
    print("\nGradient Requirements Status:")
    print("Puzzle Input requires_grad:", puzzle_tensor.requires_grad)
    
    # Check encoder path
    puzzle_emb = agent1.embedding_system.embed_puzzle(puzzle_tensor)
    print("Puzzle Embedding requires_grad:", puzzle_emb.requires_grad)
    print("Symbol Logits requires_grad:", requires_grad_info(symbol_logits1, "symbol_logits"))
    print("Symbols requires_grad:", requires_grad_info(symbols1, "symbols"))
    
    # Check decoder path
    print("Grid Logits requires_grad:", grid_logits1.requires_grad)
    print("Reconstructed requires_grad:", reconstructed1.requires_grad)
    
    # Create dot file of computation graph
    from torchviz import make_dot
    
    # Properly reshape logits and target
    batch_size, height, width, num_symbols = grid_logits1.shape
    logits_reshaped = grid_logits1.reshape(-1, num_symbols)
    target_reshaped = puzzle_tensor.reshape(-1)
    
    print("\nShape Information:")
    print(f"Grid Logits shape: {grid_logits1.shape}")
    print(f"Puzzle Tensor shape: {puzzle_tensor.shape}")
    print(f"Reshaped logits shape: {logits_reshaped.shape}")
    print(f"Reshaped target shape: {target_reshaped.shape}")
    
    loss = torch.nn.functional.cross_entropy(logits_reshaped, target_reshaped)
    
    # Compute and print gradients
    loss.backward()
    
    print("\nGradient Magnitudes:")
    def print_gradient_stats(name, parameters):
        grads = [p.grad.abs().mean().item() for p in parameters if p.grad is not None]
        if grads:
            print(f"{name}:")
            print(f"  Mean gradient magnitude: {sum(grads)/len(grads):.8f}")
            print(f"  Max gradient magnitude: {max(grads):.8f}")
            print(f"  Min gradient magnitude: {min(grads):.8f}")
    
    print_gradient_stats("Encoder", agent1.encoder.parameters())
    print_gradient_stats("Decoder", agent2.decoder.parameters())
    print_gradient_stats("Embedding System", agent1.embedding_system.parameters())
    
    # Print intermediate activations
    print("\nActivation Statistics:")
    def print_tensor_stats(name, tensor):
        if isinstance(tensor, tuple):
            for i, t in enumerate(tensor):
                print(f"{name}_{i}:")
                print(f"  Mean: {t.abs().mean().item():.6f}")
                print(f"  Max: {t.abs().max().item():.6f}")
                print(f"  Min: {t.abs().min().item():.6f}")
        else:
            print(f"{name}:")
            print(f"  Mean: {tensor.abs().mean().item():.6f}")
            print(f"  Max: {tensor.abs().max().item():.6f}")
            print(f"  Min: {tensor.abs().min().item():.6f}")
    
    print_tensor_stats("Symbol Logits", symbol_logits1)
    print_tensor_stats("Symbols", symbols1)
    print_tensor_stats("Grid Logits", grid_logits1)
    dot = make_dot(loss, params=dict(list(agent1.named_parameters()) + list(agent2.named_parameters())))
    dot.render("computation_graph", format="png")

if __name__ == "__main__":
    # Create agents and sample puzzle
    agent1 = Agent("agent1", embedding_dim=512, hidden_dim=1024)
    agent2 = Agent("agent2", embedding_dim=512, hidden_dim=1024)
    
    # Create a sample puzzle
    input_grid = np.random.randint(0, 10, size=(5, 5))
    output_grid = input_grid.copy()
    output_grid[1:4, 1:4] = np.random.randint(0, 10, size=(3, 3))
    puzzle = Puzzle(
        train_inputs=[input_grid],
        train_outputs=[output_grid],
        test_input=input_grid.copy()
    )
    
    # Convert to tensor
    puzzle_tensor = torch.tensor(
        puzzle.test_input,
        dtype=torch.long,
        device=next(agent1.parameters()).device
    ).unsqueeze(0)
    
    # Check gradient paths
    check_gradient_paths(agent1, agent2, puzzle_tensor)