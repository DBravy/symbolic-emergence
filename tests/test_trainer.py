import torch
import numpy as np
import json
import sys
import os
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agent import Agent
from puzzle import Puzzle
from trainer import CommunicationTrainer

def print_puzzle_comparison(original, agent1_recon, agent2_recon, metrics=None):
    """Print a side-by-side comparison of original and reconstructed puzzles."""
    def format_grid(grid):
        # Convert values to integers for display
        formatted_rows = []
        for row in grid:
            formatted_row = []
            for x in row:
                # Handle different types of inputs
                if isinstance(x, np.ndarray):
                    # For probability distributions or one-hot encodings,
                    # get the index of the maximum value
                    value = np.argmax(x)
                    formatted_row.append(f"{value:2d}")
                elif isinstance(x, (int, np.integer)):
                    formatted_row.append(f"{x:2d}")
                else:
                    # For float values, round to nearest integer
                    formatted_row.append(f"{int(round(x)):2d}")
            formatted_rows.append(" ".join(formatted_row))
        return "\n".join(formatted_rows)
    
    print("\n" + "="*60)
    if metrics:
        print(f"Cycle: {metrics.get('cycle', 'N/A')}")
        print(f"Total Loss: {metrics.get('total_loss', 0):.4f}")
        print(f"Grid Loss 1: {metrics.get('grid_loss1', 0):.4f}")
        print(f"Grid Loss 2: {metrics.get('grid_loss2', 0):.4f}")
        print(f"Embedding Sim 1: {metrics.get('embedding_sim1', 0):.4f}")
        print(f"Embedding Sim 2: {metrics.get('embedding_sim2', 0):.4f}")
        print(f"Message Lengths - Agent1: {metrics.get('message_length1', 0)}, Agent2: {metrics.get('message_length2', 0)}")
        print(f"Nonzero Symbols - Agent1: {metrics.get('nonzero_symbols1', 0)}, Agent2: {metrics.get('nonzero_symbols2', 0)}")
    print("="*60)
    
    # Split grids into lines
    orig_lines = format_grid(original).split('\n')
    agent1_lines = format_grid(agent1_recon).split('\n')
    agent2_lines = format_grid(agent2_recon).split('\n')
    
    # Print headers
    print(f"{'Original':^20} | {'Agent 1':^20} | {'Agent 2':^20}")
    print("-"*20 + "+" + "-"*21 + "+" + "-"*20)
    
    # Print grid rows side by side
    for orig, a1, a2 in zip(orig_lines, agent1_lines, agent2_lines):
        print(f"{orig:^20} | {a1:^20} | {a2:^20}")
    
    print("-"*63)

def visualize_puzzle_communication(trainer, puzzle_tensor, cycle=None):
    """
    Visualize the complete bidirectional communication process between agents,
    with detailed gradient tracking at each step.
    """
    print("\n" + "="*80)
    print(f"Communication Visualization (Cycle {cycle if cycle is not None else 'Initial'})")
    print("="*80)

    def print_section(title):
        print(f"\n{'-'*20} {title} {'-'*20}")

    # Store metrics for final visualization
    all_metrics = {}

    # ============ First Direction: Agent1 -> Agent2 ============
    print_section("Direction 1: Agent1 -> Agent2")

    # --- Agent 1 Encoding ---
    print("\nAgent 1 Encoding Phase:")
    puzzle_emb1 = trainer.agent1.embedding_system.embed_puzzle(puzzle_tensor)
    symbols1, symbol_logits1, length_stats1 = trainer.agent1.encode_puzzle_to_message(
        puzzle_tensor, temperature=1.0, initial_phase=False
    )

    # Debug print
    print(f"Type of length_stats1: {type(length_stats1)}")
    print(f"Content of length_stats1: {length_stats1}")

    print(f"Message stats from Agent 1:")
    print(f"- Message length: {length_stats1['total_length']}")
    print(f"- Nonzero symbols: {length_stats1['nonzero_symbols']}")
    print(f"- Unique symbols used: {torch.unique(symbols1).tolist()}")
    
    # --- Agent 2 Decoding ---
    print("\nAgent 2 Decoding Phase:")
    reconstructed1, grid_logits1 = trainer.agent2.decode_message_to_puzzle(
        symbols1,
        target_size=(puzzle_tensor.size(1), puzzle_tensor.size(2))
    )
    
    # Compute loss for first direction
    grid_loss1 = F.cross_entropy(
        grid_logits1.reshape(-1, trainer.agent2.puzzle_symbols),
        puzzle_tensor.reshape(-1)
    )
    
    # ============ Second Direction: Agent2 -> Agent1 ============
    print_section("Direction 2: Agent2 -> Agent1")
    
    # --- Agent 2 Encoding ---
    print("\nAgent 2 Encoding Phase:")
    puzzle_emb2 = trainer.agent2.embedding_system.embed_puzzle(puzzle_tensor)
    symbols2, symbol_logits2, length_stats2 = trainer.agent2.encode_puzzle_to_message(
        puzzle_tensor, temperature=1.0, initial_phase=False
    )
    
    # --- Agent 1 Decoding ---
    print("\nAgent 1 Decoding Phase:")
    reconstructed2, grid_logits2 = trainer.agent1.decode_message_to_puzzle(
        symbols2,
        target_size=(puzzle_tensor.size(1), puzzle_tensor.size(2))
    )
    
    # Compute loss for second direction
    grid_loss2 = F.cross_entropy(
        grid_logits2.reshape(-1, trainer.agent1.puzzle_symbols),
        puzzle_tensor.reshape(-1)
    )

    # ============ Compute Total Loss and Backward Pass ============
    print_section("Loss Computation and Backward Pass")
    
    total_loss = grid_loss1 + grid_loss2
    total_loss.backward()
    
    # ============ Gradient Analysis ============
    print_section("Gradient Analysis After Backward Pass")
    
    def analyze_gradients(agent, agent_name):
        print(f"\n{agent_name} Gradients:")
        
        # print("\nEncoder Gradients:")
        # for name, param in agent.encoder.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}:")
        #         print(f"  - Gradient norm: {grad_norm:.6f}")
        #         print(f"  - Requires grad: {param.requires_grad}")
        #     else:
        #         print(f"{name}:")
        #         print(f"  - No gradient")
        #         print(f"  - Requires grad: {param.requires_grad}")
        
        # print("\nDecoder Gradients:")
        # for name, param in agent.decoder.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}:")
        #         print(f"  - Gradient norm: {grad_norm:.6f}")
        #         print(f"  - Requires grad: {param.requires_grad}")
        #     else:
        #         print(f"{name}:")
        #         print(f"  - No gradient")
        #         print(f"  - Requires grad: {param.requires_grad}")

    analyze_gradients(trainer.agent1, "Agent 1")
    analyze_gradients(trainer.agent2, "Agent 2")

    # ============ Reconstruction Visualization ============
    print_section("Puzzle Reconstruction Comparison")
    
    all_metrics.update({
        'cycle': cycle,
        'total_loss': total_loss.item(),
        'grid_loss1': grid_loss1.item(),
        'grid_loss2': grid_loss2.item(),
        'message_length1': length_stats1['total_length'],
        'message_length2': length_stats2['total_length'],
        'nonzero_symbols1': length_stats1['nonzero_symbols'],
        'nonzero_symbols2': length_stats2['nonzero_symbols']
    })
    
    # Detach tensors before converting to numpy
    print_puzzle_comparison(
        original=puzzle_tensor[0].cpu().numpy(),
        agent1_recon=reconstructed1[0].detach().cpu().numpy(),
        agent2_recon=reconstructed2[0].detach().cpu().numpy(),
        metrics=all_metrics
    )

    return all_metrics

def load_arc_puzzles(file_path):
    """Load all examples from ARC puzzles JSON file."""
    # Get the directory where the test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to the JSON file
    json_path = os.path.join(current_dir, file_path)
    print(f"Looking for file at: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_examples = []
    for puzzle_id, puzzle_data in data.items():
        try:
            # Extract all training examples
            for train_example in puzzle_data['train']:
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(train_example['input']),
                        np.array(train_example['output'])
                    )
                )
                
                # Also use the output as an input example since we're testing communication
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(train_example['output']),
                        np.array(train_example['output'])
                    )
                )
            
            # Extract test examples
            for test_example in puzzle_data['test']:
                # For test examples we have input but may not have output
                # Use input as both input and output since we're testing communication
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(test_example['input']),
                        np.array(test_example['input'])
                    )
                )
                
                # If test example has output, use that too
                if 'output' in test_example:
                    all_examples.append(
                        Puzzle.from_single_example(
                            np.array(test_example['output']),
                            np.array(test_example['output'])
                        )
                    )
                    
        except (ValueError, TypeError) as e:
            print(f"Skipping puzzle {puzzle_id} due to error: {e}")
            continue
            
    print(f"Extracted {len(all_examples)} total examples from {len(data)} puzzles")
    return all_examples

def test_embedding_convergence_with_vis():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    embedding_dim = 128
    hidden_dim = 256
    num_symbols = 15
    puzzle_symbols = 10
    max_seq_length = 5
    max_grid_size = (30, 30)

    # Create two agents
    agent1 = Agent(
        agent_id='agent1', 
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        max_grid_size=max_grid_size
    )
    agent1 = agent1.to(device)

    agent2 = Agent(
        agent_id='agent2', 
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        max_grid_size=max_grid_size
    )
    agent2 = agent2.to(device)

    # Create trainer
    trainer = CommunicationTrainer(
        agent1=agent1, 
        agent2=agent2, 
        learning_rate=1e-3
    )

    # Load puzzles
    puzzles = load_arc_puzzles('arc-agi_test_challenges.json')
    print(f"Loaded {len(puzzles)} puzzles from ARC dataset")
    puzzles = puzzles[:5]  # Limit to first 5 puzzles

    # Metrics logging
    metrics_file = 'training_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'step', 'total_loss', 'grid_loss1', 'grid_loss2', 
            'embedding_sim1', 'embedding_sim2', 
            'message_length1', 'message_length2',
            'nonzero_symbols1', 'nonzero_symbols2'
        ])
        writer.writeheader()

    # Training parameters
    num_cycles = 10000
    num_exchanges = 1

    # Training loop
    all_metrics = []
    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}/{num_cycles}")
        
        for puzzle_idx, puzzle in enumerate(puzzles):
            # Convert puzzle to tensor
            puzzle_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0)
            
            # Visualize initial state before training
            if cycle == 0 and puzzle_idx == 0:
                print("\nInitial Puzzle Reconstruction:")
                visualize_puzzle_communication(trainer, puzzle_tensor)
            
            # Perform training step
            cycle_metrics = trainer.train_bidirectional_step(
                puzzle_tensor, 
                num_exchanges=num_exchanges,
                temperature=1.0,
                initial_phase=False
            )
            
            # Append training metrics
            all_metrics.extend(cycle_metrics)
            
            # Visualize after training
            print(f"\nPuzzle {puzzle_idx + 1} Reconstruction after Cycle {cycle + 1}:")
            visualize_puzzle_communication(trainer, puzzle_tensor, cycle)
            
            # Log metrics to CSV
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'step', 'total_loss', 'grid_loss1', 'grid_loss2', 
                    'embedding_sim1', 'embedding_sim2', 
                    'message_length1', 'message_length2',
                    'nonzero_symbols1', 'nonzero_symbols2'
                ])
                for metric in cycle_metrics:
                    writer.writerow({
                        'step': len(all_metrics),
                        'total_loss': metric.get('total_loss', 0),
                        'grid_loss1': metric.get('grid_loss1', 0),
                        'grid_loss2': metric.get('grid_loss2', 0),
                        'embedding_sim1': metric.get('embedding_sim1', 0),
                        'embedding_sim2': metric.get('embedding_sim2', 0),
                        'message_length1': metric.get('message_length1', 0),
                        'message_length2': metric.get('message_length2', 0),
                        'nonzero_symbols1': metric.get('nonzero_symbols1', 0),
                        'nonzero_symbols2': metric.get('nonzero_symbols2', 0)
                    })

    # Create final plot
    create_final_plot(metrics_file)

def create_final_plot(metrics_file: str):
    """Create a comprehensive plot using all stored metrics."""
    # Read all data from CSV
    steps = []
    total_losses = []
    grid_losses1 = []
    grid_losses2 = []
    
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            total_losses.append(float(row['total_loss']))
            grid_losses1.append(float(row['grid_loss1']))
            grid_losses2.append(float(row['grid_loss2']))
    
    # Create final plot with three subplots
    plt.figure(figsize=(12, 12))
    
    # Plot total loss with smoothed curve
    plt.subplot(3, 1, 1)
    plt.plot(steps, total_losses, label='Total Loss', alpha=0.3)
    # Add smoothed line using rolling average
    window_size = 20
    smoothed_losses = np.convolve(total_losses, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = steps[window_size-1:]
    plt.plot(smoothed_steps, smoothed_losses, label='Smoothed Total Loss', linewidth=2)
    plt.title('Total Loss Over Training')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot grid losses
    plt.subplot(3, 1, 2)
    plt.plot(steps, grid_losses1, label='Grid Loss Agent 1', alpha=0.7)
    plt.plot(steps, grid_losses2, label='Grid Loss Agent 2', alpha=0.7)
    plt.title('Grid Losses Over Training')
    plt.xlabel('Training Step')
    plt.ylabel('Grid Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss ratio (grid loss / total loss)
    plt.subplot(3, 1, 3)
    grid_loss_total = np.array(grid_losses1) + np.array(grid_losses2)
    loss_ratio = grid_loss_total / np.array(total_losses)
    plt.plot(steps, loss_ratio, label='Grid Loss / Total Loss Ratio', color='purple')
    plt.title('Grid Loss to Total Loss Ratio')
    plt.xlabel('Training Step')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('final_training_plot.png')
    plt.close()

if __name__ == "__main__":
    test_embedding_convergence_with_vis()