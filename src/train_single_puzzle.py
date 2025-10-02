"""
Single-Puzzle ARC Training Script

This script trains two agents to communicate about a single ARC puzzle using only one symbol.
The agents learn to abstract the underlying pattern by communicating the example pairs back and forth.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import os
import sys
from datetime import datetime

# Import existing components
from agent_selection import ProgressiveSelectionAgent as Agent
from trainer_selection import ProgressiveSelectionTrainer as Trainer
from puzzle import Puzzle


def load_arc_puzzle_by_id(puzzle_id: str, arc_file='arc-agi_test_challenges.json'):
    """Load a specific ARC puzzle by ID"""
    with open(arc_file, 'r') as f:
        data = json.load(f)
    
    if puzzle_id not in data:
        raise ValueError(f"Puzzle ID '{puzzle_id}' not found in dataset")
    
    puzzle_data = data[puzzle_id]
    print(f"\nLoaded puzzle: {puzzle_id}")
    print(f"  Training examples: {len(puzzle_data['train'])}")
    print(f"  Test examples: {len(puzzle_data['test'])}")
    
    return puzzle_data, puzzle_id


def create_puzzle_pairs(puzzle_data):
    """Convert ARC puzzle to input-output pairs for communication training"""
    pairs = []
    
    # Use all training examples as communication pairs
    for train_ex in puzzle_data['train']:
        input_grid = np.array(train_ex['input'])
        output_grid = np.array(train_ex['output'])
        pairs.append(Puzzle.from_single_example(input_grid, output_grid))
        print(f"  - Training pair: input {input_grid.shape} -> output {output_grid.shape}")
    
    # Test example
    test_input = np.array(puzzle_data['test'][0]['input'])
    test_output = None
    if 'output' in puzzle_data['test'][0]:
        test_output = np.array(puzzle_data['test'][0]['output'])
    
    print(f"  - Test: input {test_input.shape}", end='')
    if test_output is not None:
        print(f" -> output {test_output.shape}")
    else:
        print()
    
    return pairs, test_input, test_output


def train_single_puzzle(
    puzzle_id: str,
    cycles: int = 100,
    status_file: str = 'training_status.json',
    output_dir: str = './outputs',
    arc_file: str = 'arc-agi_test_challenges.json',
    device: str = None
):
    """Train agents on a single ARC puzzle with one communication symbol"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Single-Puzzle ARC Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Puzzle ID: {puzzle_id}")
    print(f"Training cycles: {cycles}")
    
    # Update status
    def update_status(status, message, progress=0):
        status_data = {
            'status': status,
            'message': message,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        }
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
    
    update_status('loading', 'Loading ARC puzzle...', 0)
    
    # Load puzzle
    puzzle_data, puzzle_id = load_arc_puzzle_by_id(puzzle_id, arc_file)
    training_pairs, test_input, test_output = create_puzzle_pairs(puzzle_data)
    
    if len(training_pairs) == 0:
        raise ValueError("No training pairs found in puzzle")
    
    print(f"\nCreated {len(training_pairs)} training pairs")
    
    # Architecture configuration - minimal for single symbol
    config = {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_symbols': 11,  # 10 puzzle symbols + 1 communication symbol
        'puzzle_symbols': 10,
        'max_seq_length': 1,  # Only one position
        'learning_rate': 1e-5,
        'num_distractors': 2
    }
    
    print(f"\nArchitecture:")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Puzzle symbols: {config['puzzle_symbols']}")
    print(f"  Communication symbols: 1 (single symbol)")
    print(f"  Sequence length: 1")
    
    update_status('initializing', 'Creating agents...', 5)
    
    # Create agents
    agent1 = Agent(
        agent_id="agent1",
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_symbols=config['num_symbols'],
        puzzle_symbols=config['puzzle_symbols'],
        max_seq_length=config['max_seq_length'],
        similarity_metric='cosine'
    ).to(device)
    
    agent2 = Agent(
        agent_id="agent2",
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_symbols=config['num_symbols'],
        puzzle_symbols=config['puzzle_symbols'],
        max_seq_length=config['max_seq_length'],
        similarity_metric='cosine'
    ).to(device)
    
    # Initialize agents to use only 1 communication symbol
    agent1.current_comm_symbols = 1
    agent2.current_comm_symbols = 1
    agent1.current_seq_length = 1
    agent2.current_seq_length = 1
    
    print(f"\nAgents created:")
    print(f"  Agent 1: {sum(p.numel() for p in agent1.parameters())} parameters")
    print(f"  Agent 2: {sum(p.numel() for p in agent2.parameters())} parameters")
    
    update_status('initializing', 'Creating trainer...', 10)
    
    # Create trainer
    trainer = Trainer(
        agent1=agent1,
        agent2=agent2,
        learning_rate=config['learning_rate'],
        device=device,
        num_distractors=config['num_distractors'],
        distractor_strategy='random',
        first_training_cycles=cycles,
        training_cycles=cycles,
        web_mode=True
    )
    
    # Set the training pairs as active puzzles
    trainer.active_puzzles = training_pairs
    trainer.available_arc_puzzles = training_pairs
    
    print(f"\nStarting training loop...")
    print(f"Training on {len(training_pairs)} example pairs from puzzle {puzzle_id}")
    
    # Training loop - Use reconstruction training
    print(f"\nUsing reconstruction training mode")
    
    for cycle in range(cycles):
        progress = int((cycle / cycles) * 90) + 10
        update_status('training', f'Cycle {cycle+1}/{cycles}', progress)
        
        cycle_losses = []
        cycle_recon_accs = []
        
        # Train on each example pair using reconstruction
        for idx, pair in enumerate(training_pairs):
            # Convert input and output to tensors
            input_tensor = torch.tensor(
                pair.test_input,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)
            
            output_tensor = torch.tensor(
                pair.test_output,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)
            
            # Bidirectional reconstruction training
            agent1.train()
            agent2.train()
            trainer.opt1.zero_grad()
            trainer.opt2.zero_grad()
            
            # Direction 1: Agent1 encodes input -> Agent2 decodes to reconstruct output
            symbols1, _, _ = agent1.encode_puzzle_to_message(input_tensor, temperature=1.0, initial_phase=False)
            
            # Convert symbols to embeddings for decoder
            current_comm_embeddings = agent2.communication_embedding.weight[
                config['puzzle_symbols']:config['puzzle_symbols'] + 1
            ]
            embedded_msg = torch.matmul(symbols1, current_comm_embeddings)
            
            # Decode WITHOUT forcing size - let decoder learn to predict it
            decoded_logits, _, _, (height_logits, width_logits) = agent2.decoder(
                embedded_msg, temperature=1.0, input_grid=input_tensor
            )
            
            # Compute reconstruction loss on overlapping region
            B, Hp, Wp, C = decoded_logits.shape
            Ht, Wt = output_tensor.shape[1], output_tensor.shape[2]
            Hc, Wc = min(Hp, Ht), min(Wp, Wt)
            
            recon_loss1 = nn.functional.cross_entropy(
                decoded_logits[:, :Hc, :Wc, :].reshape(B * Hc * Wc, C),
                output_tensor[:, :Hc, :Wc].reshape(B * Hc * Wc)
            )
            
            # Add size prediction loss
            height_target = torch.tensor([max(1, min(Ht, agent2.decoder.max_height)) - 1], device=device)
            width_target = torch.tensor([max(1, min(Wt, agent2.decoder.max_width)) - 1], device=device)
            size_loss1 = nn.functional.cross_entropy(height_logits, height_target) + \
                            nn.functional.cross_entropy(width_logits, width_target)
            
            loss1 = recon_loss1 + 0.1 * size_loss1
            
            # Direction 2: Agent2 encodes input -> Agent1 decodes to reconstruct output
            symbols2, _, _ = agent2.encode_puzzle_to_message(input_tensor, temperature=1.0, initial_phase=False)
            
            current_comm_embeddings = agent1.communication_embedding.weight[
                config['puzzle_symbols']:config['puzzle_symbols'] + 1
            ]
            embedded_msg2 = torch.matmul(symbols2, current_comm_embeddings)
            
            # Decode WITHOUT forcing size
            decoded_logits2, _, _, (height_logits2, width_logits2) = agent1.decoder(
                embedded_msg2, temperature=1.0, input_grid=input_tensor
            )
            
            # Compute reconstruction loss on overlapping region
            B2, Hp2, Wp2, C2 = decoded_logits2.shape
            Hc2, Wc2 = min(Hp2, Ht), min(Wp2, Wt)
            
            recon_loss2 = nn.functional.cross_entropy(
                decoded_logits2[:, :Hc2, :Wc2, :].reshape(B2 * Hc2 * Wc2, C2),
                output_tensor[:, :Hc2, :Wc2].reshape(B2 * Hc2 * Wc2)
            )
            
            # Add size prediction loss
            size_loss2 = nn.functional.cross_entropy(height_logits2, height_target) + \
                            nn.functional.cross_entropy(width_logits2, width_target)
            
            loss2 = recon_loss2 + 0.1 * size_loss2
            
            # Total loss
            total_loss = loss1 + loss2
            total_loss.backward()
            
            trainer.opt1.step()
            trainer.opt2.step()
            
            cycle_losses.append(total_loss.item())
            
            # Compute reconstruction accuracy on overlapping regions
            with torch.no_grad():
                pred1 = decoded_logits[:, :Hc, :Wc, :].argmax(dim=-1)
                pred2 = decoded_logits2[:, :Hc2, :Wc2, :].argmax(dim=-1)
                acc1 = (pred1 == output_tensor[:, :Hc, :Wc]).float().mean().item()
                acc2 = (pred2 == output_tensor[:, :Hc2, :Wc2]).float().mean().item()
                cycle_recon_accs.append((acc1 + acc2) / 2)
        
        # Log progress
        if cycle % 10 == 0:
            avg_loss = np.mean(cycle_losses) if cycle_losses else 0
            avg_acc = np.mean(cycle_recon_accs) if cycle_recon_accs else 0
            print(f"Cycle {cycle:4d}: Loss={avg_loss:.4f}, Recon_Acc={avg_acc:.3f}")
    
    update_status('testing', 'Evaluating on test example...', 95)
    
    # Test on the test input
# Test on the test input
    print(f"\n{'='*60}")
    print(f"Testing on test example with reconstruction...")
    print(f"{'='*60}")
    
    test_tensor = torch.tensor(test_input, dtype=torch.long, device=device).unsqueeze(0)
    
    # Have agent1 encode and agent2 decode to reconstruct
    reconstructed_output = None
    accuracy = None
    with torch.no_grad():
        agent1.eval()
        agent2.eval()
        
        # Agent1 encodes test input
        symbols, _, _ = agent1.encode_puzzle_to_message(test_tensor, temperature=0.1)
        print(f"Agent 1 encoded test input to symbol: {torch.argmax(symbols[0, 0, :]).item() - config['puzzle_symbols']}")
        
        # Agent2 decodes to reconstruct
        current_comm_embeddings = agent2.communication_embedding.weight[
            config['puzzle_symbols']:config['puzzle_symbols'] + 1
        ]
        embedded_msg = torch.matmul(symbols, current_comm_embeddings)
        
        # Decode WITHOUT forcing size - let decoder predict naturally
        decoded_logits, _, _, (height_logits, width_logits) = agent2.decoder(
            embedded_msg, temperature=0.1, input_grid=test_tensor
        )
        
        # Get predicted size
        pred_height = height_logits.argmax(dim=-1).item() + 1
        pred_width = width_logits.argmax(dim=-1).item() + 1
        
        reconstructed_output = decoded_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        
        print(f"Agent 2 reconstructed output with shape: {reconstructed_output.shape}")
        print(f"Decoder predicted size: {pred_height}x{pred_width}")
        if test_output is not None:
            print(f"Expected output size: {test_output.shape[0]}x{test_output.shape[1]}")
        
        # Compute accuracy if we have the expected output
        if test_output is not None:
            test_output_tensor = torch.tensor(test_output, dtype=torch.long, device=device)
            reconstructed_tensor = torch.tensor(reconstructed_output, dtype=torch.long, device=device)
            
            # Handle size mismatches - compare overlapping region
            min_h = min(reconstructed_tensor.shape[0], test_output_tensor.shape[0])
            min_w = min(reconstructed_tensor.shape[1], test_output_tensor.shape[1])
            
            accuracy = (reconstructed_tensor[:min_h, :min_w] == test_output_tensor[:min_h, :min_w]).float().mean().item()
            print(f"Reconstruction accuracy (overlapping region): {accuracy:.3f}")
            print(f"Correct pixels: {int(accuracy * min_h * min_w)}/{min_h * min_w}")
            print(f"Overlapping region: {min_h}x{min_w}")
    
    # Update status with reconstruction result (this must be the FINAL status write)
    final_status = {
        'status': 'complete',
        'message': f'Training complete on puzzle {puzzle_id}',
        'progress': 100,
        'timestamp': datetime.now().isoformat(),
        'reconstruction': {
            'test_input': test_input.tolist(),
            'test_output': test_output.tolist() if test_output is not None else None,
            'reconstructed': reconstructed_output.tolist() if reconstructed_output is not None else None,
            'accuracy': accuracy if test_output is not None else None
        }
    }
    
    with open(status_file, 'w') as f:
        json.dump(final_status, f)
    
    print(f"\nReconstruction result saved to status file")
    # DO NOT call update_status() here - it would overwrite the reconstruction data!
    
    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'single_puzzle_{puzzle_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    
    torch.save({
        'puzzle_id': puzzle_id,
        'puzzle_data': puzzle_data,
        'config': config,
        'agent1_state': agent1.state_dict(),
        'agent2_state': agent2.state_dict(),
        'cycles': cycles
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train agents on a single ARC puzzle')
    parser.add_argument('--puzzle-id', type=str, required=True, help='ARC puzzle ID to train on')
    parser.add_argument('--cycles', type=int, default=100, help='Number of training cycles')
    parser.add_argument('--arc-file', type=str, default='arc-agi_test_challenges.json', 
                        help='Path to ARC challenges JSON file')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--status-file', type=str, default='training_status.json', 
                        help='Status file for web interface')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_single_puzzle(
        puzzle_id=args.puzzle_id,
        cycles=args.cycles,
        status_file=args.status_file,
        output_dir=args.output_dir,
        arc_file=args.arc_file,
        device=args.device
    )


if __name__ == '__main__':
    main()

