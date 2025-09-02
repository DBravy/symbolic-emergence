import torch
import matplotlib.pyplot as plt
from puzzle import Puzzle
import numpy as np
import json
import os
import torch.nn.functional as F
from collections import deque

# Import our new retrieval classes
from retrieval_agent import RetrievalAgent
from retrieval_trainer import RetrievalCommunicationTrainer

# Import existing utility classes and functions
from train import (MovingAverage, load_arc_puzzles, plot_training_metrics, 
                  print_communication_debug, print_message_details, print_grid,
                  print_symbol_consistency)

def prepare_puzzle_database(arc_puzzles, max_puzzles=50):
    """
    Prepare puzzle database from ARC puzzles for retrieval training
    
    Args:
        arc_puzzles: List of Puzzle objects
        max_puzzles: Maximum number of puzzles to include
        
    Returns:
        List of torch tensors representing puzzles
    """
    print(f"Preparing puzzle database from {len(arc_puzzles)} ARC puzzles...")
    
    puzzle_tensors = []
    used_puzzles = set()  # Track puzzle hashes to avoid duplicates
    
    for puzzle in arc_puzzles[:max_puzzles * 2]:  # Sample more than needed to filter
        try:
            # Get puzzle array
            puzzle_array = puzzle.test_input
            
            # Convert to hashable tuple to check for duplicates
            puzzle_hash = tuple(tuple(row) for row in puzzle_array)
            if puzzle_hash in used_puzzles:
                continue
            
            # Filter puzzles by size (reasonable limits)
            height, width = puzzle_array.shape
            if height < 2 or width < 2 or height > 15 or width > 15:
                continue
            
            # Convert to tensor
            puzzle_tensor = torch.tensor(puzzle_array, dtype=torch.long)
            puzzle_tensors.append(puzzle_tensor)
            used_puzzles.add(puzzle_hash)
            
            if len(puzzle_tensors) >= max_puzzles:
                break
                
        except Exception as e:
            print(f"Skipping puzzle due to error: {e}")
            continue
    
    print(f"Created database with {len(puzzle_tensors)} unique puzzles")
    
    # Print some statistics
    sizes = [(p.shape[0], p.shape[1]) for p in puzzle_tensors]
    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    
    print(f"Size statistics:")
    print(f"  Heights: {min(heights)}-{max(heights)} (avg: {np.mean(heights):.1f})")
    print(f"  Widths: {min(widths)}-{max(widths)} (avg: {np.mean(widths):.1f})")
    
    return puzzle_tensors

# Remove unused classes since we're using the detailed approach from train.py
def evaluate_phase_transition(trainer, test_puzzles, phase_name):
    """Evaluate agent performance at phase transitions (simplified version)"""
    print(f"\n--- Evaluating {phase_name} Phase Performance ---")
    
    # Get training stats
    stats = trainer.get_training_stats()
    print(f"Training Statistics:")
    print(f"  Current phase: {stats['current_phase']}")
    print(f"  Total cycles: {stats['cycle_count']}")
    print(f"  Retrieval cycles: {stats['retrieval_cycles']}")
    print(f"  Generation cycles: {stats['generation_cycles']}")
    
    # Evaluate retrieval accuracy if in retrieval mode
    if stats['current_phase'] == 'retrieval' and len(trainer.puzzle_database) > 0:
        # Convert puzzle objects to tensors for evaluation
        test_tensors = []
        for puzzle in test_puzzles[:10]:  # Evaluate on subset
            try:
                tensor = torch.tensor(puzzle.test_input, dtype=torch.long)
                test_tensors.append(tensor)
            except:
                continue
        
        if test_tensors:
            accuracy_results = trainer.evaluate_retrieval_accuracy(test_tensors, num_samples=10)
            return accuracy_results
    
    return None

def train_single_phase_cycle(cycle, trainer, test_puzzles, device, phase_name, puzzle_to_symbols=None):
    """Train for one cycle in either phase with detailed logging like original train.py"""
    cycle_metrics = []
    
    visualization_frequency = 10  # Match original train.py
    
    for puzzle_idx, puzzle in enumerate(test_puzzles):
        puzzle_tensor = torch.tensor(
            puzzle.test_input, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        step_metrics = trainer.train_bidirectional_step(
            puzzle_tensor, 
            num_exchanges=1,
            temperature=1.0,
            initial_phase=False
        )
        
        cycle_metrics.extend(step_metrics)
        
        # Add phase information
        for metrics in step_metrics:
            metrics['training_phase'] = phase_name
        
        # Show communication debug with vocabulary info (same as original train.py)
        if cycle % visualization_frequency == 0 and puzzle_idx == 0:  # Only show for first puzzle
            print(f"\n--- Visualization for {phase_name.title()} Cycle {cycle}, Puzzle {puzzle_idx} ---")
            print_communication_debug(puzzle_tensor, trainer.agent1, trainer.agent2)
        
        # Track symbol usage for consistency checking (same as original train.py)
        if puzzle_to_symbols is not None and trainer.training_mode in ["encoder_only", "joint"]:
            for metrics in step_metrics:
                if 'encoder1_symbol' in metrics and 'encoder2_symbol' in metrics:
                    puzzle_id = f"puzzle_{puzzle_idx}"
                    if puzzle_id not in puzzle_to_symbols:
                        puzzle_to_symbols[puzzle_id] = {
                            'encoder1': [],
                            'encoder2': []
                        }
                    
                    puzzle_to_symbols[puzzle_id]['encoder1'].append(metrics['encoder1_symbol'])
                    puzzle_to_symbols[puzzle_id]['encoder2'].append(metrics['encoder2_symbol'])
    
    return cycle_metrics

def plot_two_phase_metrics(retrieval_metrics, generation_metrics, vocab_history=None):
    """Plot metrics for both training phases"""
    
    # Combine metrics for plotting
    all_metrics = retrieval_metrics + generation_metrics
    
    # Create phase indicators
    retrieval_end = len(retrieval_metrics)
    
    plt.figure(figsize=(20, 15))
    
    # Plot loss
    plt.subplot(4, 1, 1)
    losses = [m['total_loss'] for m in all_metrics if not np.isnan(m['total_loss'])]
    plt.plot(losses, label='Loss')
    plt.axvline(x=retrieval_end, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Two-Phase Training - Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot size accuracies
    plt.subplot(4, 1, 2)
    size_acc1 = [m.get('agent1_size_accuracy', 0) for m in all_metrics]
    size_acc2 = [m.get('agent2_size_accuracy', 0) for m in all_metrics]
    plt.plot(size_acc1, label='Agent1 Size Acc', alpha=0.8)
    plt.plot(size_acc2, label='Agent2 Size Acc', alpha=0.8)
    plt.axvline(x=retrieval_end, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Two-Phase Training - Size Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot content accuracies
    plt.subplot(4, 1, 3)
    content_acc1 = [m.get('agent1_content_accuracy', 0) for m in all_metrics]
    content_acc2 = [m.get('agent2_content_accuracy', 0) for m in all_metrics]
    plt.plot(content_acc1, label='Agent1 Content Acc', alpha=0.7)
    plt.plot(content_acc2, label='Agent2 Content Acc', alpha=0.7)
    plt.axvline(x=retrieval_end, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Two-Phase Training - Content Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot phase information
    plt.subplot(4, 1, 4)
    phases = [1 if m.get('training_phase') == 'retrieval' else 2 for m in all_metrics]
    plt.plot(phases, linewidth=2, label='Training Phase')
    plt.axvline(x=retrieval_end, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Training Phase')
    plt.xlabel('Step')
    plt.ylabel('Phase (1=Retrieval, 2=Generation)')
    plt.ylim(0.5, 2.5)
    plt.yticks([1, 2], ['Retrieval', 'Generation'])
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('two_phase_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training function with detailed logging like original train.py"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ARC puzzles
    arc_file_path = 'arc-agi_test_challenges.json'
    test_puzzles = load_arc_puzzles(arc_file_path)
    print(f"\nLoaded {len(test_puzzles)} total examples from ARC dataset")
    
    # Select diverse puzzles for training
    selected_puzzles = [test_puzzles[i] for i in range(0, len(test_puzzles), 10)][:20]
    print(f"Selected {len(selected_puzzles)} diverse puzzles for training")
    
    # Prepare puzzle database for retrieval phase
    puzzle_database = prepare_puzzle_database(selected_puzzles, max_puzzles=30)
    
    # Create retrieval-capable agents
    sender = RetrievalAgent(
        agent_id="retrieval_sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=14,      # Maximum symbols
        puzzle_symbols=10,
        max_seq_length=2,    # Maximum sequence length
        sender_scale=1.0,
        puzzle_database=puzzle_database
    ).to(device)
    
    receiver = RetrievalAgent(
        agent_id="retrieval_receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=14,      # Maximum symbols
        puzzle_symbols=10,
        max_seq_length=2,    # Maximum sequence length
        sender_scale=1.0,
        puzzle_database=puzzle_database
    ).to(device)
    
    # Create retrieval trainer
    trainer = RetrievalCommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=1e-4,
        device=device,
        expansion_frequency=10000,  # Match original train.py
        symbols_per_expansion=2,   # Add 2 symbols per expansion
        length_per_expansion=1,    # Add 1 to sequence length per expansion
        puzzle_database=puzzle_database
    )
    
    # Show initial state (same as original train.py)
    print("\n" + "="*60)
    print("INITIAL VOCABULARY STATE")
    print("="*60)
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    
    # Show retrieval stats
    print("\nRetrieval System Status:")
    sender_stats = sender.get_retrieval_stats()
    receiver_stats = receiver.get_retrieval_stats()
    print(f"  Sender: {sender_stats}")
    print(f"  Receiver: {receiver_stats}")
    print("="*60)
    
    # Initialize histories and trackers (same structure as original train.py)
    all_metrics_history = []
    acc1_size_history = []
    acc1_content_history = []
    acc2_size_history = []
    acc2_content_history = []
    acc1_size_binary_history = [] 
    acc2_size_binary_history = [] 
    
    # Track vocabulary progression (same as original train.py)
    vocab_history = {
        'cycles': [],
        'vocab_sizes': [],
        'seq_lengths': [],
        'expansion_cycles': []
    }
    
    # Initialize moving averages (same as original train.py)
    ma_window = 50
    acc1_size_ma = MovingAverage(ma_window)
    acc1_content_ma = MovingAverage(ma_window)
    acc2_size_ma = MovingAverage(ma_window)
    acc2_content_ma = MovingAverage(ma_window)
    
    # Track symbol consistency (same as original train.py)
    puzzle_to_symbols = {}
    
    # Enhanced logging with detailed console output
    log_file_path = 'two_phase_detailed_training_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write("Two-Phase Communication Training Log (Detailed)\n")
        log_file.write("="*60 + "\n")
        log_file.write("Phase 1: Retrieval - Agents select from existing puzzles\n")
        log_file.write("Phase 2: Generation - Agents generate puzzles from scratch\n")
        log_file.write(f"Expansion every {trainer.expansion_frequency} cycles\n")
        log_file.write(f"Add {trainer.symbols_per_expansion} symbols and {trainer.length_per_expansion} sequence length per expansion\n")
        log_file.write("Size Accuracy: Using distance-based metric (1.0 = perfect, decreases with distance)\n")
        log_file.write("="*60 + "\n\n")
        
        # PHASE 1: RETRIEVAL TRAINING
        retrieval_cycles = 500  # Increased for better learning
        print(f"\n{'='*80}")
        print(f"PHASE 1: RETRIEVAL TRAINING ({retrieval_cycles} cycles)")
        print(f"{'='*80}")
        
        log_file.write(f"STARTING RETRIEVAL PHASE - {retrieval_cycles} cycles\n")
        log_file.write(f"Database size: {len(puzzle_database)} puzzles\n")
        log_file.write("-" * 50 + "\n")
        
        # Set trainer to joint mode and retrieval phase
        trainer.set_training_mode("joint")
        trainer.set_training_phase("retrieval")
        
        cycle_offset = 0
        
        for cycle in range(retrieval_cycles):
            total_cycle = cycle_offset + cycle + 1
            print(f"\nRetrieval Cycle {cycle + 1}/{retrieval_cycles} (Total: {total_cycle})")
            
            # Track vocabulary progression (same frequency as original train.py)
            if cycle == 0 or cycle % 20 == 0:
                vocab_history['cycles'].append(total_cycle)
                vocab_history['vocab_sizes'].append(sender.current_comm_symbols)
                vocab_history['seq_lengths'].append(sender.current_seq_length)
            
            # Check for expansion events
            if cycle > 0 and cycle % trainer.expansion_frequency == 0:
                vocab_history['expansion_cycles'].append(total_cycle)
                log_file.write(f"VOCABULARY EXPANSION at cycle {total_cycle}\n")
                log_file.write(f"  New vocab size: {sender.current_comm_symbols}\n")
                log_file.write(f"  New seq length: {sender.current_seq_length}\n")
                log_file.write("-" * 40 + "\n")
            
            # Train one cycle
            cycle_metrics = train_single_phase_cycle(
                cycle, trainer, selected_puzzles, device, "retrieval", puzzle_to_symbols
            )
            all_metrics_history.extend(cycle_metrics)
            
            # Update histories and moving averages (same as original train.py)
            for metrics in cycle_metrics:
                acc1_size_ma.update(metrics['agent1_size_accuracy'])
                acc1_content_ma.update(metrics['agent1_content_accuracy'])
                acc2_size_ma.update(metrics['agent2_size_accuracy'])
                acc2_content_ma.update(metrics['agent2_content_accuracy'])
                
                # Store history
                acc1_size_history.append(acc1_size_ma.get_average())
                acc1_content_history.append(acc1_content_ma.get_average())
                acc2_size_history.append(acc2_size_ma.get_average())
                acc2_content_history.append(acc2_content_ma.get_average())
                
                # Binary accuracy for comparison
                acc1_size_binary_history.append(metrics['agent1_size_binary'])
                acc2_size_binary_history.append(metrics['agent2_size_binary'])
            
            # Enhanced logging (same format as original train.py)
            avg_metrics = {
                'total_loss': np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])]),
                'avg_size_error1': np.mean([m.get('agent1_size_error', 0) for m in cycle_metrics]),
                'avg_size_error2': np.mean([m.get('agent2_size_error', 0) for m in cycle_metrics]),
                'training_phase': 'retrieval'
            }
            
            log_file.write(
                f"Cycle {total_cycle}: " + 
                f"Phase={avg_metrics['training_phase']}, " +
                f"Loss={avg_metrics['total_loss']:.4f}, " +
                f"Vocab={sender.current_comm_symbols}, " +
                f"SeqLen={sender.current_seq_length}, " +
                f"Size_Acc1={acc1_size_ma.get_average():.3f}, " +
                f"Content_Acc1={acc1_content_ma.get_average():.3f}, " +
                f"Size_Acc2={acc2_size_ma.get_average():.3f}, " +
                f"Content_Acc2={acc2_content_ma.get_average():.3f}, " +
                f"SizeErr1={avg_metrics['avg_size_error1']:.1f}, " +
                f"SizeErr2={avg_metrics['avg_size_error2']:.1f}\n"
            )
            log_file.flush()
            
            # Plot metrics periodically (same frequency as original train.py)
            if (cycle + 1) % 10 == 0:
                accuracy_histories = {
                    'acc1_size': acc1_size_history,
                    'acc1_content': acc1_content_history,
                    'acc2_size': acc2_size_history,
                    'acc2_content': acc2_content_history,
                    'acc1_size_binary': acc1_size_binary_history,
                    'acc2_size_binary': acc2_size_binary_history
                }
                
                plot_training_metrics(all_metrics_history, accuracy_histories, vocab_history, 
                                    title=f"Retrieval Training Metrics (Cycle {total_cycle})")
            
            # Evaluate retrieval accuracy periodically
            if (cycle + 1) % 50 == 0:
                test_tensors = puzzle_database[:10]  # Evaluate on subset
                accuracy_results = trainer.evaluate_retrieval_accuracy(test_tensors, num_samples=10)
                log_file.write(f"Retrieval Accuracy at cycle {total_cycle}: {accuracy_results['average_retrieval_accuracy']:.3f}\n")
        
        # Print symbol consistency after retrieval phase
        print_symbol_consistency(puzzle_to_symbols)
        
        # PHASE 2: GENERATION TRAINING
        generation_cycles = 500
        cycle_offset = retrieval_cycles
        
        print(f"\n{'='*80}")
        print(f"PHASE 2: GENERATION TRAINING ({generation_cycles} cycles)")
        print(f"{'='*80}")
        
        log_file.write(f"\nSTARTING GENERATION PHASE - {generation_cycles} cycles\n")
        log_file.write("-" * 50 + "\n")
        
        # Switch to generation mode
        trainer.set_training_phase("generation")
        
        for cycle in range(generation_cycles):
            total_cycle = cycle_offset + cycle + 1
            print(f"\nGeneration Cycle {cycle + 1}/{generation_cycles} (Total: {total_cycle})")
            
            # Track vocabulary progression
            if cycle % 20 == 0:
                vocab_history['cycles'].append(total_cycle)
                vocab_history['vocab_sizes'].append(sender.current_comm_symbols)
                vocab_history['seq_lengths'].append(sender.current_seq_length)
            
            # Check for expansion events
            if cycle > 0 and cycle % trainer.expansion_frequency == 0:
                vocab_history['expansion_cycles'].append(total_cycle)
                log_file.write(f"VOCABULARY EXPANSION at cycle {total_cycle}\n")
                log_file.write(f"  New vocab size: {sender.current_comm_symbols}\n")
                log_file.write(f"  New seq length: {sender.current_seq_length}\n")
                log_file.write("-" * 40 + "\n")
            
            # Train one cycle
            cycle_metrics = train_single_phase_cycle(
                cycle, trainer, selected_puzzles, device, "generation", puzzle_to_symbols
            )
            all_metrics_history.extend(cycle_metrics)
            
            # Update histories and moving averages
            for metrics in cycle_metrics:
                acc1_size_ma.update(metrics['agent1_size_accuracy'])
                acc1_content_ma.update(metrics['agent1_content_accuracy'])
                acc2_size_ma.update(metrics['agent2_size_accuracy'])
                acc2_content_ma.update(metrics['agent2_content_accuracy'])
                
                # Store history
                acc1_size_history.append(acc1_size_ma.get_average())
                acc1_content_history.append(acc1_content_ma.get_average())
                acc2_size_history.append(acc2_size_ma.get_average())
                acc2_content_history.append(acc2_content_ma.get_average())
                
                # Binary accuracy for comparison
                acc1_size_binary_history.append(metrics['agent1_size_binary'])
                acc2_size_binary_history.append(metrics['agent2_size_binary'])
            
            # Enhanced logging
            avg_metrics = {
                'total_loss': np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])]),
                'avg_size_error1': np.mean([m.get('agent1_size_error', 0) for m in cycle_metrics]),
                'avg_size_error2': np.mean([m.get('agent2_size_error', 0) for m in cycle_metrics]),
                'training_phase': 'generation'
            }
            
            log_file.write(
                f"Cycle {total_cycle}: " + 
                f"Phase={avg_metrics['training_phase']}, " +
                f"Loss={avg_metrics['total_loss']:.4f}, " +
                f"Vocab={sender.current_comm_symbols}, " +
                f"SeqLen={sender.current_seq_length}, " +
                f"Size_Acc1={acc1_size_ma.get_average():.3f}, " +
                f"Content_Acc1={acc1_content_ma.get_average():.3f}, " +
                f"Size_Acc2={acc2_size_ma.get_average():.3f}, " +
                f"Content_Acc2={acc2_content_ma.get_average():.3f}, " +
                f"SizeErr1={avg_metrics['avg_size_error1']:.1f}, " +
                f"SizeErr2={avg_metrics['avg_size_error2']:.1f}\n"
            )
            log_file.flush()
            
            # Plot metrics periodically
            if (cycle + 1) % 10 == 0:
                accuracy_histories = {
                    'acc1_size': acc1_size_history,
                    'acc1_content': acc1_content_history,
                    'acc2_size': acc2_size_history,
                    'acc2_content': acc2_content_history,
                    'acc1_size_binary': acc1_size_binary_history,
                    'acc2_size_binary': acc2_size_binary_history
                }
                
                plot_training_metrics(all_metrics_history, accuracy_histories, vocab_history, 
                                    title=f"Two-Phase Training Metrics (Cycle {total_cycle})")
    
    # Final detailed summary (same as original train.py)
    print("\n" + "="*60)
    print("FINAL TRAINING SUMMARY")
    print("="*60)

    # Show recent size errors
    recent_metrics = all_metrics_history[-50:]  # Last 50 steps
    if recent_metrics:
        avg_size_error1 = np.mean([m.get('agent1_size_error', 0) for m in recent_metrics])
        avg_size_error2 = np.mean([m.get('agent2_size_error', 0) for m in recent_metrics])
        
        print(f"Average Size Error (last 50 steps):")
        print(f"  Agent 1: {avg_size_error1:.2f} units")
        print(f"  Agent 2: {avg_size_error2:.2f} units")
        
        final_size_acc1 = acc1_size_ma.get_average()
        final_size_acc2 = acc2_size_ma.get_average()
        final_content_acc1 = acc1_content_ma.get_average()
        final_content_acc2 = acc2_content_ma.get_average()
        
        print(f"\nFinal Accuracies (distance-based):")
        print(f"  Agent 1 - Size: {final_size_acc1:.3f}, Content: {final_content_acc1:.3f}")
        print(f"  Agent 2 - Size: {final_size_acc2:.3f}, Content: {final_content_acc2:.3f}")

    # Show final vocabulary state
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    
    # Final symbol consistency check
    print_symbol_consistency(puzzle_to_symbols)
    
    print("="*60)
    print(f"\nTwo-phase training complete! Check {log_file_path} for detailed logs")
    print("Plots saved as training_metrics.png")

if __name__ == "__main__":
    main()