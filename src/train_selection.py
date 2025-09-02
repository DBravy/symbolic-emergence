import torch
# SELECTION MODIFICATION: Use selection versions
from agent_selection import ProgressiveSelectionAgent as Agent
from trainer_selection import ProgressiveSelectionTrainer as CommunicationTrainer

import matplotlib.pyplot as plt
from puzzle import Puzzle
import numpy as np
import json
import os
import torch.nn.functional as F
from collections import deque
import random

class MovingAverage:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def update(self, value):
        self.values.append(value)
        
    def get_average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

# Existing functions from train.py (unchanged)
def load_arc_puzzles(file_path):
    """Load all examples from ARC puzzles JSON file."""
    with open(file_path, 'r') as f:
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
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(test_example['input']),
                        np.array(test_example['input'])
                    )
                )
                
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

def run_pretraining_phase(trainer, target_puzzles=None, epochs=50):
    """
    Modified pretraining to only train on puzzles that have symbol mappings
    NOW USES GLOBAL INDICES CONSISTENTLY
    """
    print(f"\n{'='*60}")
    print(f"PRETRAINING PHASE - Encoder Training ({epochs} epochs)")
    print(f"{'='*60}")
    
    if target_puzzles is None:
        target_puzzles = trainer.active_puzzles
    
    agent1, agent2 = trainer.agent1, trainer.agent2
    device = trainer.device
    
    # Filter target puzzles to only include those with symbol mappings
    mapped_puzzles = []
    mapped_puzzle_indices = []
    
    for i, puzzle in enumerate(target_puzzles):
        # Find this puzzle's index in active_puzzles
        try:
            active_idx = trainer.active_puzzles.index(puzzle)
            if active_idx in trainer.puzzle_symbol_mapping:
                mapped_puzzles.append(puzzle)
                mapped_puzzle_indices.append(active_idx)
        except ValueError:
            # Puzzle not in active list - skip
            continue
    
    print(f"Training on {len(mapped_puzzles)} puzzles with symbol mappings")
    print(f"Skipping {len(target_puzzles) - len(mapped_puzzles)} puzzles without mappings")
    
    if len(mapped_puzzles) == 0:
        print("No puzzles with symbol mappings to train on!")
        return {'loss': [], 'accuracy': [], 'epochs': []}
    
    # Show current vocabulary state
    print(f"\nCurrent Vocabulary State:")
    print(f"  Agent1 communication symbols: {agent1.current_comm_symbols}")
    print(f"  Agent1 total symbols: {agent1.current_total_symbols}")
    print(f"  Puzzle symbols range: 0-{agent1.puzzle_symbols-1}")
    print(f"  Communication symbols range: {agent1.puzzle_symbols}-{agent1.current_total_symbols-1}")
    
    # Show existing symbol mappings
    print(f"\nCurrent Puzzle-Symbol Mappings:")
    for puzzle_idx, symbol_idx in trainer.puzzle_symbol_mapping.items():
        print(f"  Puzzle {puzzle_idx} → Symbol {symbol_idx}")
    
    # Convert puzzles to tensors and assign target symbols
    # NOW USING GLOBAL INDICES AS KEYS
    puzzle_tensors = {}  # global_idx -> tensor
    targets = {}         # global_idx -> target_symbol
    global_to_local = {} # global_idx -> position in processing order
    
    print(f"\nPuzzle-Symbol Assignments for Pretraining:")
    for local_pos, (puzzle, global_idx) in enumerate(zip(mapped_puzzles, mapped_puzzle_indices)):
        puzzle_tensor = torch.tensor(
            puzzle.test_input, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        # Store using global index
        puzzle_tensors[global_idx] = puzzle_tensor
        global_to_local[global_idx] = local_pos
        
        # Get the symbol mapping
        target_symbol = trainer.puzzle_symbol_mapping[global_idx]
        comm_symbol_idx = target_symbol - agent1.puzzle_symbols
        targets[global_idx] = comm_symbol_idx
        
        print(f"  Global Puzzle {global_idx}: Symbol {target_symbol} (Comm #{comm_symbol_idx})")
        print(f"    Grid shape: {puzzle_tensor.shape[1:]} - {puzzle_tensor[0].cpu().numpy()[:3, :3]}...")
    
    print(f"\nSymbol Assignment Summary:")
    print(f"  Puzzles with symbol mappings: {len(mapped_puzzles)}")
    print(f"  Global indices: {sorted(mapped_puzzle_indices)}")
    print(f"  Training mapping: {dict(sorted(targets.items()))}")
    
    # Show what we're training
    print(f"\nTraining Configuration:")
    print(f"  Encoder components: encoder, embedding_system, message_pooling")
    print(f"  Target symbols: {sorted(list(set(targets.values())))}")
    print(f"  Agent2 disabled during pretraining")
    
    # Training setup
    encoder_params = []
    component_counts = {'encoder': 0, 'embedding_system': 0, 'message_pooling': 0}
    
    for name, param in agent1.named_parameters():
        if any(component in name for component in ['encoder', 'embedding_system', 'message_pooling']):
            encoder_params.append(param)
            param.requires_grad = True
            
            # Count parameters by component
            for component in component_counts:
                if component in name:
                    component_counts[component] += param.numel()
        else:
            param.requires_grad = False
    
    print(f"\nTrainable Parameters:")
    for component, count in component_counts.items():
        print(f"  {component}: {count:,} parameters")
    print(f"  Total trainable: {sum(component_counts.values()):,} parameters")
    
    # Disable Agent2 during pretraining
    for param in agent2.parameters():
        param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=0.001)
    
    history = {
        'loss': [],
        'accuracy': [],
        'epochs': []
    }
    
    visualization_frequency = max(1, epochs // 5)  # Show details 5 times during training
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        epoch_predictions = {}  # Track what each puzzle predicted
        
        # Create randomized order of global indices
        global_indices = list(mapped_puzzle_indices)
        random.shuffle(global_indices)
        
        show_details = (epoch % visualization_frequency == 0) or (epoch == epochs - 1)
        if show_details:
            print(f"\n--- Pretraining Epoch {epoch+1} Details ---")
        
        for global_idx in global_indices:
            puzzle_tensor = puzzle_tensors[global_idx]
            
            encoder_optimizer.zero_grad()
            
            # Forward pass through Agent1's encoder
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1, deterministic=True
            )
            
            pred_symbol = symbols[0, 0].argmax().item()
            target = torch.tensor([targets[global_idx]], device=device)
            
            # Store prediction for visualization
            epoch_predictions[global_idx] = pred_symbol
            
            # Symbol prediction loss
            symbol_loss = F.cross_entropy(symbol_logits[0, 0].unsqueeze(0), target)
            
            # Regularization
            reg_loss = 0.0
            for param in encoder_params:
                reg_loss += param.pow(2.0).sum()
            
            total_loss_item = symbol_loss + 0.001 * reg_loss
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            encoder_optimizer.step()
            
            # Track metrics
            if pred_symbol == targets[global_idx]:
                correct += 1
            total_loss += total_loss_item.item()
            
            # Show detailed learning for first few puzzles during visualization epochs
            if show_details and global_to_local[global_idx] < min(3, len(mapped_puzzles)):
                correct_symbol = "✓" if pred_symbol == targets[global_idx] else "✗"
                confidence = F.softmax(symbol_logits[0, 0], dim=0)[targets[global_idx]].item()
                print(f"  Global Puzzle {global_idx}: Target {targets[global_idx]} → Predicted {pred_symbol} {correct_symbol} (conf: {confidence:.3f})")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(mapped_puzzles)
        accuracy = correct / len(mapped_puzzles)
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['epochs'].append(epoch + 1)
        
        if show_details:
            print(f"  Epoch {epoch+1} Summary: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
            
            # Show symbol prediction distribution
            symbol_counts = {}
            for global_puzzle_idx, pred in epoch_predictions.items():
                if pred not in symbol_counts:
                    symbol_counts[pred] = []
                symbol_counts[pred].append(global_puzzle_idx)
            
            print(f"  Symbol predictions this epoch:")
            for symbol in sorted(symbol_counts.keys()):
                global_puzzles = symbol_counts[symbol]
                correct_count = sum(1 for p in global_puzzles if targets[p] == symbol)
                print(f"    Symbol {symbol}: {len(global_puzzles)} puzzles ({correct_count} correct) - global puzzles {sorted(global_puzzles)}")
        
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Pretraining Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
    
    # Final symbol assignment verification
    print(f"\n{'='*40}")
    print(f"PRETRAINING VERIFICATION")
    print(f"{'='*40}")
    
    agent1.eval()
    with torch.no_grad():
        print(f"Final symbol assignments (verification):")
        for global_idx in sorted(mapped_puzzle_indices):
            puzzle_tensor = puzzle_tensors[global_idx]
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1, deterministic=True
            )
            pred_symbol = symbols[0, 0].argmax().item()
            target_symbol = targets[global_idx]
            confidence = F.softmax(symbol_logits[0, 0], dim=0)[target_symbol].item()
            
            status = "✓" if pred_symbol == target_symbol else "✗"
            print(f"  Global Puzzle {global_idx}: Target {target_symbol} → Final {pred_symbol} {status} (conf: {confidence:.3f})")
    
    agent1.train()
    
    # Copy Agent1's weights to Agent2
    print(f"\n{'='*40}")
    print(f"COPYING WEIGHTS Agent1 → Agent2")
    print(f"{'='*40}")
    
    with torch.no_grad():
        total_copied = 0
        
        # Copy encoder
        encoder_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.encoder.named_parameters(), agent2.encoder.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                encoder_copied += param1.numel()
        print(f"  ✓ Encoder: {encoder_copied:,} parameters")
        total_copied += encoder_copied
        
        # Copy embedding system
        embedding_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.embedding_system.named_parameters(), agent2.embedding_system.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                embedding_copied += param1.numel()
        print(f"  ✓ Embedding system: {embedding_copied:,} parameters")
        total_copied += embedding_copied
        
        # Copy message pooling
        pooling_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.message_pooling.named_parameters(), agent2.message_pooling.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                pooling_copied += param1.numel()
        print(f"  ✓ Message pooling: {pooling_copied:,} parameters")
        total_copied += pooling_copied
        
        # Copy communication embeddings
        start_idx = agent1.puzzle_symbols
        end_idx = start_idx + agent1.current_comm_symbols
        agent2.communication_embedding.weight[start_idx:end_idx].copy_(
            agent1.communication_embedding.weight[start_idx:end_idx]
        )
        comm_copied = agent1.current_comm_symbols * agent1.communication_embedding.embedding_dim
        print(f"  ✓ Communication embeddings: {comm_copied:,} parameters (symbols {start_idx}-{end_idx-1})")
        total_copied += comm_copied
        
        print(f"Total parameters copied: {total_copied:,}")
    
    # Re-enable all parameters
    for param in agent1.parameters():
        param.requires_grad = True
    for param in agent2.parameters():
        param.requires_grad = True
    
    print(f"\n{'='*40}")
    print(f"PRETRAINING COMPLETE")
    print(f"{'='*40}")
    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Trained on {len(mapped_puzzles)} puzzles with symbol mappings")
    print(f"Global puzzle indices: {sorted(mapped_puzzle_indices)}")
    print(f"Both agents now have synchronized encoders")
    
    return history

def run_training_phase(trainer, cycles=200):
    """Run the main training phase"""
    print(f"\n{'='*60}")
    print(f"TRAINING PHASE - Joint Training ({cycles} cycles)")
    print(f"{'='*60}")
    
    trainer.set_training_mode("joint")
    
    # Initialize tracking
    metrics_history = []
    acc1_selection_history = []
    acc2_selection_history = []
    conf1_correct_history = []
    conf2_correct_history = []
    
    # Moving averages
    ma_window = 20
    acc1_selection_ma = MovingAverage(ma_window)
    acc2_selection_ma = MovingAverage(ma_window)
    conf1_correct_ma = MovingAverage(ma_window)
    conf2_correct_ma = MovingAverage(ma_window)
    
    for cycle in range(cycles):
        print(f"\nTraining Cycle {cycle + 1}/{cycles}")
        
        # Train on each active puzzle
        cycle_metrics = []
        for puzzle_idx, puzzle in enumerate(trainer.active_puzzles):
            puzzle_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=trainer.device
            ).unsqueeze(0)
            
            step_metrics = trainer.train_bidirectional_step(
                puzzle_tensor, 
                puzzle_idx,
                num_exchanges=1,
                temperature=1.0,
                initial_phase=False
            )
            
            cycle_metrics.extend(step_metrics)
        
        # Update metrics
        metrics_history.extend(cycle_metrics)
        
        for metrics in cycle_metrics:
            acc1_selection_ma.update(metrics['agent1_selection_accuracy'])
            acc2_selection_ma.update(metrics['agent2_selection_accuracy'])
            conf1_correct_ma.update(metrics['agent1_correct_confidence'])
            conf2_correct_ma.update(metrics['agent2_correct_confidence'])
            
            acc1_selection_history.append(acc1_selection_ma.get_average())
            acc2_selection_history.append(acc2_selection_ma.get_average())
            conf1_correct_history.append(conf1_correct_ma.get_average())
            conf2_correct_history.append(conf2_correct_ma.get_average())
        
        # Show progress
        if cycle_metrics:
            avg_acc1 = acc1_selection_ma.get_average()
            avg_acc2 = acc2_selection_ma.get_average()
            avg_loss = np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])])
            print(f"  Avg Loss: {avg_loss:.4f}, Acc1: {avg_acc1:.3f}, Acc2: {avg_acc2:.3f}")
    
    accuracies_history = {
        'acc1_selection': acc1_selection_history,
        'acc2_selection': acc2_selection_history,
        'conf1_correct': conf1_correct_history,
        'conf2_correct': conf2_correct_history
    }
    
    return metrics_history, accuracies_history

def run_consolidation_phase(trainer):
    """Run consolidation phase to remove recessive symbols"""
    print(f"\n{'='*60}")
    print(f"CONSOLIDATION PHASE")
    print(f"{'='*60}")
    
    # Run consolidation tests
    confusion_data = trainer.run_consolidation_test()
    
    # Identify recessive symbols
    recessive_symbols = trainer.identify_recessive_symbols(confusion_data)
    
    # Remove recessive symbols
    trainer.remove_recessive_symbols(recessive_symbols)
    
    return confusion_data, recessive_symbols

def run_addition_phase(trainer):
    """Run addition phase to add new puzzles"""
    print(f"\n{'='*60}")
    print(f"ADDITION PHASE")
    print(f"{'='*60}")
    
    new_puzzles = trainer.add_new_puzzles()
    return new_puzzles

# Updated plotting function for phase-based training
def plot_phase_training_metrics(metrics_history, accuracies_history, phase_info, title="Phase-Based Training Metrics"):
    """Plot training metrics for phase-based training"""
    plt.figure(figsize=(20, 12))
    
    # Separate metrics by phase
    phases = ['pretraining', 'training', 'consolidation', 'addition']
    phase_colors = {'pretraining': 'blue', 'training': 'green', 'consolidation': 'orange', 'addition': 'red'}
    
    # Plot loss
    plt.subplot(3, 2, 1)
    losses = [m['total_loss'] for m in metrics_history if not np.isnan(m['total_loss'])]
    phase_markers = [m.get('phase', 'training') for m in metrics_history if not np.isnan(m['total_loss'])]
    
    current_phase = None
    start_idx = 0
    
    for i, phase in enumerate(phase_markers + ['END']):
        if phase != current_phase or i == len(phase_markers):
            if current_phase is not None and start_idx < i:
                plt.plot(range(start_idx, i), losses[start_idx:i], 
                        label=f'{current_phase.title()} Loss', 
                        color=phase_colors.get(current_phase, 'gray'),
                        alpha=0.7)
            current_phase = phase
            start_idx = i
    
    plt.title(f'{title} - Loss by Phase')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot selection accuracies
    plt.subplot(3, 2, 2)
    plt.plot(accuracies_history['acc1_selection'], label='Agent1 Selection Acc', alpha=0.8, linewidth=2)
    plt.plot(accuracies_history['acc2_selection'], label='Agent2 Selection Acc', alpha=0.8, linewidth=2)
    plt.title(f'{title} - Selection Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot confidence
    plt.subplot(3, 2, 3)
    plt.plot(accuracies_history['conf1_correct'], label='Agent1 Correct Confidence', alpha=0.7)
    plt.plot(accuracies_history['conf2_correct'], label='Agent2 Correct Confidence', alpha=0.7)
    plt.title(f'{title} - Confidence in Correct Selection')
    plt.xlabel('Step')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot vocabulary size over time
    plt.subplot(3, 2, 4)
    vocab_sizes = [m.get('active_puzzles', 0) for m in metrics_history]
    global_phases = [m.get('global_phase_count', 0) for m in metrics_history]
    
    plt.plot(vocab_sizes, label='Active Puzzles/Symbols', linewidth=2, color='purple')
    plt.title('Vocabulary Evolution')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    
    # Phase information summary
    plt.subplot(3, 2, 5)
    phase_text = f"Current Phase: {phase_info.get('current_phase', 'unknown').title()}\n"
    phase_text += f"Phase Cycle: {phase_info.get('phase_cycle', 0)}\n"
    phase_text += f"Global Phase Count: {phase_info.get('global_phase_count', 0)}\n"
    phase_text += f"Active Puzzles: {phase_info.get('active_puzzles', 0)}\n"
    phase_text += f"Removed Symbols: {phase_info.get('removed_symbols', 0)}\n\n"
    
    phase_text += "Phase Cycle:\n"
    phase_text += "1. Pretraining (new puzzles)\n"
    phase_text += "2. Training (all puzzles)\n"
    phase_text += "3. Consolidation (remove recessive)\n"
    phase_text += "4. Addition (add new puzzles)\n"
    
    plt.text(0.1, 0.9, phase_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Phase Information')
    
    # Training configuration
    plt.subplot(3, 2, 6)
    config = phase_info.get('selection_config', {})
    config_text = f"Training Configuration:\n\n"
    config_text += f"Num Distractors: {config.get('num_distractors', 'N/A')}\n"
    config_text += f"Distractor Strategy: {config.get('distractor_strategy', 'N/A')}\n"
    config_text += f"Training Cycles: {config.get('training_cycles', 'N/A')}\n"
    config_text += f"Consolidation Tests: {config.get('consolidation_tests', 'N/A')}\n"
    config_text += f"Puzzles per Addition: {config.get('puzzles_per_addition', 'N/A')}\n"
    
    plt.text(0.1, 0.9, config_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Configuration')
    
    plt.tight_layout()
    plt.savefig('phase_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_selection_debug(puzzle_tensor, sender, receiver, trainer):
    """Debug function for selection task - updated for phase training"""
    print("\n  Phase-Based Selection Debug:")
    print_grid(puzzle_tensor[0], "Target Puzzle")
    
    # Show phase and vocabulary status
    phase_info = trainer.get_phase_status()
    print(f"\nCurrent Phase: {phase_info['current_phase']}")
    print(f"Phase Cycle: {phase_info['phase_cycle']}")
    print(f"Active Puzzles: {phase_info['active_puzzles']}")
    print(f"Removed Symbols: {phase_info['removed_symbols']}")
    
    # Show current puzzle-symbol mapping
    print(f"\nPuzzle-Symbol Mapping:")
    for puzzle_idx, symbol_idx in trainer.puzzle_symbol_mapping.items():
        print(f"  Puzzle {puzzle_idx} → Symbol {symbol_idx}")
    
    # Sender encodes message
    print(f"\nSender → Receiver Selection:")
    symbols, symbol_logits, stats = sender.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols, "Sender")
    
    # Create selection candidates from active puzzles
    candidates = []
    for puzzle in trainer.active_puzzles[:min(4, len(trainer.active_puzzles))]:  # Limit for display
        candidate_tensor = torch.tensor(
            puzzle.test_input, 
            dtype=torch.long, 
            device=puzzle_tensor.device
        ).unsqueeze(0)
        candidates.append(candidate_tensor)
    
    # Receiver selects
    selection_probs, selection_logits, debug_info = receiver.select_from_candidates(
        symbols, candidates, temperature=0.1
    )
    
    # Show results
    predicted_idx = selection_logits.argmax(dim=-1).item()
    target_confidence = selection_probs[0, 0].item()
    
    print(f"\nReceiver Selection Results:")
    print(f"  Number of candidates: {len(candidates)}")
    print(f"  Predicted choice: {predicted_idx} (target is 0)")
    print(f"  Selection correct: {'✓' if predicted_idx == 0 else '✗'}")
    print(f"  Confidence in target: {target_confidence:.4f}")
    
    print(f"\nAll selection probabilities:")
    for i, prob in enumerate(selection_probs[0]):
        marker = " ← target" if i == 0 else f" ← candidate {i}"
        symbol = "✓" if i == predicted_idx else " "
        print(f"    {symbol} Candidate {i}: {prob.item():.4f}{marker}")

def print_grid(grid: torch.Tensor, title: str = "Grid"):
    """Print a grid in a readable format"""
    print(f"\n{title}:")
    for row in grid.cpu().numpy():
        print("  " + " ".join(f"{x:2d}" for x in row))

def print_message_details(symbols: torch.Tensor, agent_name: str):
    """Print detailed information about the message being sent."""
    if len(symbols.shape) == 3:  # [batch, seq, num_symbols]
        message_indices = torch.argmax(symbols, dim=-1)[0]
    else:
        message_indices = symbols[0]
        
    nonzero_indices = message_indices[message_indices != 0]
    
    print(f"\n{agent_name} Message:")
    print(f"  Full sequence: {message_indices.tolist()}")
    print(f"  Non-zero symbols: {nonzero_indices.tolist()}")
    print(f"  Length: {len(message_indices)}")
    print(f"  Active symbols: {len(nonzero_indices)}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create selection agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=100,     # Increased to accommodate growth
        puzzle_symbols=10,
        max_seq_length=1,    
        sender_scale=1.0,
        similarity_metric='cosine'
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=100,     # Increased to accommodate growth
        puzzle_symbols=10,
        max_seq_length=1,    
        sender_scale=1.0,
        similarity_metric='cosine'
    ).to(device)
    
    # Create phase-based trainer
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=1e-7,
        device=device,
        sync_frequency=50,
        num_distractors=4,
        distractor_strategy='random',
        training_cycles=200,         # Cycles per training phase
        consolidation_tests=5,       # Test rounds in consolidation
        puzzles_per_addition=5       # Puzzles to add each cycle
    )
    
    # Load ARC puzzles
    arc_file_path = 'arc-agi_test_challenges.json'
    all_puzzles = load_arc_puzzles(arc_file_path)
    print(f"\nLoaded {len(all_puzzles)} total examples from ARC dataset")
    
    # Set puzzle dataset and initialize first puzzles
    trainer.set_puzzle_dataset(all_puzzles)
    trainer.initialize_first_puzzles(initial_count=5)
    
    # Show initial state
    print("\n" + "="*60)
    print("INITIAL STATE")
    print("="*60)
    phase_info = trainer.get_phase_status()
    print(f"Phase: {phase_info['current_phase']}")
    print(f"Active puzzles: {phase_info['active_puzzles']}")
    print(f"Puzzle-symbol mapping: {phase_info['puzzle_symbol_mapping']}")
    sender.print_position_symbol_mapping()
    print("="*60)
    
    # Enhanced logging
    with open('phase_training_log.txt', 'w') as log_file:
        log_file.write("Phase-Based Training Log\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Phase cycle: pretraining → training → consolidation → addition\n")
        log_file.write(f"Training cycles per phase: {trainer.training_cycles}\n")
        log_file.write(f"Consolidation tests: {trainer.consolidation_tests}\n")
        log_file.write(f"Puzzles per addition: {trainer.puzzles_per_addition}\n")
        log_file.write("="*50 + "\n\n")
        
        # Initialize comprehensive tracking
        all_metrics_history = []
        all_accuracies_history = {
            'acc1_selection': [],
            'acc2_selection': [],
            'conf1_correct': [],
            'conf2_correct': []
        }
        
        # Main phase cycle loop
        max_global_phases = 3  # Run 3 complete cycles for demonstration
        
        while trainer.global_phase_count < max_global_phases:
            phase_info = trainer.get_phase_status()
            current_phase = phase_info['current_phase']
            
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"GLOBAL PHASE {trainer.global_phase_count + 1} - {current_phase.upper()}\n")
            log_file.write(f"{'='*60}\n")
            log_file.flush()
            
            if current_phase == "pretraining":
                # Pretraining phase - train encoder on newly added puzzles
                if trainer.global_phase_count == 0:
                    # First pretraining - use all initial puzzles
                    pretraining_history = run_pretraining_phase(trainer, epochs=150)
                else:
                    # Subsequent pretraining - use only newly added puzzles
                    # Get last 5 puzzles (newly added)
                    new_puzzles = trainer.active_puzzles[-trainer.puzzles_per_addition:]
                    pretraining_history = run_pretraining_phase(trainer, target_puzzles=new_puzzles, epochs=30)
                
                trainer.advance_phase()
                
            elif current_phase == "training":
                # Training phase - full joint training
                training_metrics, training_accuracies = run_training_phase(trainer, cycles=trainer.training_cycles)
                
                # Add to comprehensive tracking
                all_metrics_history.extend(training_metrics)
                for key in all_accuracies_history:
                    all_accuracies_history[key].extend(training_accuracies[key])
                
                # Log training summary
                if training_metrics:
                    final_acc1 = training_accuracies['acc1_selection'][-1] if training_accuracies['acc1_selection'] else 0
                    final_acc2 = training_accuracies['acc2_selection'][-1] if training_accuracies['acc2_selection'] else 0
                    avg_loss = np.mean([m['total_loss'] for m in training_metrics[-50:] if not np.isnan(m['total_loss'])])
                    
                    log_file.write(f"Training completed:\n")
                    log_file.write(f"  Final Agent1 accuracy: {final_acc1:.3f}\n")
                    log_file.write(f"  Final Agent2 accuracy: {final_acc2:.3f}\n")
                    log_file.write(f"  Average loss (last 50): {avg_loss:.4f}\n")
                    log_file.flush()
                
                trainer.advance_phase()
                
            elif current_phase == "consolidation":
                # Consolidation phase - test and remove recessive symbols
                confusion_data, removed_symbols = run_consolidation_phase(trainer)
                
                log_file.write(f"Consolidation completed:\n")
                log_file.write(f"  Tested symbols: {len(confusion_data)}\n")
                log_file.write(f"  Removed symbols: {len(removed_symbols)}\n")
                log_file.write(f"  Remaining puzzles: {len(trainer.active_puzzles)}\n")
                if removed_symbols:
                    log_file.write(f"  Removed: {removed_symbols}\n")
                log_file.flush()
                
                trainer.advance_phase()
                
            elif current_phase == "addition":
                # Addition phase - add new puzzles
                new_puzzles = run_addition_phase(trainer)
                
                log_file.write(f"Addition completed:\n")
                log_file.write(f"  Added puzzles: {len(new_puzzles)}\n")
                log_file.write(f"  Total active puzzles: {len(trainer.active_puzzles)}\n")
                log_file.flush()
                
                trainer.advance_phase()
            
            # Plot progress after each phase
            if all_metrics_history:
                plot_phase_training_metrics(
                    all_metrics_history, 
                    all_accuracies_history,
                    trainer.get_phase_status(),
                    title=f"Phase-Based Training (Global Phase {trainer.global_phase_count})"
                )
            
            # Show debug info periodically
            if current_phase == "training" and len(trainer.active_puzzles) > 0:
                print(f"\n--- Phase Debug Info ---")
                puzzle = trainer.active_puzzles[0]
                puzzle_tensor = torch.tensor(
                    puzzle.test_input, 
                    dtype=torch.long, 
                    device=device
                ).unsqueeze(0)
                print_selection_debug(puzzle_tensor, sender, receiver, trainer)
            
            # Safety check to prevent infinite loops
            if trainer.global_phase_count >= max_global_phases:
                break
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL PHASE-BASED TRAINING SUMMARY")
    print("="*60)
    
    final_phase_info = trainer.get_phase_status()
    print(f"Completed global phases: {final_phase_info['global_phase_count']}")
    print(f"Final active puzzles: {final_phase_info['active_puzzles']}")
    print(f"Total removed symbols: {final_phase_info['removed_symbols']}")
    print(f"Final puzzle-symbol mapping: {final_phase_info['puzzle_symbol_mapping']}")
    
    if all_metrics_history:
        recent_metrics = all_metrics_history[-50:]  # Last 50 steps
        final_acc1 = np.mean([m['agent1_selection_accuracy'] for m in recent_metrics])
        final_acc2 = np.mean([m['agent2_selection_accuracy'] for m in recent_metrics])
        print(f"\nFinal Performance:")
        print(f"  Agent 1 Selection Accuracy: {final_acc1:.3f}")
        print(f"  Agent 2 Selection Accuracy: {final_acc2:.3f}")
    
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)
    
    print("\nPhase-based training complete! Check phase_training_log.txt for details")

if __name__ == "__main__":
    main()