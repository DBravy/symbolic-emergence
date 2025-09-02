"""
Easy-to-use dynamic symbol and sequence length training script with positional constraints.
Each position in the sequence gets its own dedicated symbol vocabulary.
Modify the EXPERIMENT_CONFIG at the top to change plateau detection and symbol settings.
"""

import torch
from agent import Agent
from trainer import CommunicationTrainer
import matplotlib.pyplot as plt
from puzzle import Puzzle
import numpy as np
import json

# Import our dynamic symbol management modules (now with positional constraints)
from dynamic_symbols import DynamicTrainingManager
from dynamic_config import DynamicTrainingConfig, PresetConfigs, create_custom_config

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# Choose one of these configuration options:

# Option 1: Use a preset configuration
# EXPERIMENT_CONFIG = PresetConfigs.conservative()  # Slow, careful symbol addition
# EXPERIMENT_CONFIG = PresetConfigs.aggressive()    # Fast symbol addition
# EXPERIMENT_CONFIG = PresetConfigs.research()      # Balanced for research
# EXPERIMENT_CONFIG = PresetConfigs.minimal_start() # Start with 2 comm symbols, length 1

# Option 2: Create a custom configuration with positional constraints
EXPERIMENT_CONFIG = create_custom_config(
    plateau_cycles=100,                    # Check for plateau over 100 cycles  
    plateau_threshold_percent=20.0,        # 20% accuracy range = plateau
    min_cycles_before_detection=100,       # Wait 100 cycles before first check
    initial_comm_symbols=2,                # Start with 2 communication symbols (position 0)
    max_comm_symbols=10,                   # Allow up to 10 communication symbols (5 positions total)
    initial_seq_length=1,                  # Start with sequence length 1
    max_seq_length=5,                      # Max sequence length 5 (positions 0,1,2,3,4)
    total_training_cycles=1000             # Train for 1000 cycles total (shorter for testing)
)

# Option 3: Load from a JSON file (uncomment to use)
# EXPERIMENT_CONFIG = DynamicTrainingConfig.load_from_json("my_config.json")

# =============================================================================
# TRAINING IMPLEMENTATION
# =============================================================================

from collections import deque

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

def load_arc_puzzles(file_path):
    """Load puzzles from ARC dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_examples = []
    for puzzle_id, puzzle_data in data.items():
        try:
            for train_example in puzzle_data['train']:
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(train_example['input']),
                        np.array(train_example['output'])
                    )
                )
        except (ValueError, TypeError) as e:
            print(f"Skipping puzzle {puzzle_id} due to error: {e}")
            continue
            
    print(f"Loaded {len(all_examples)} examples from {len(data)} puzzles")
    return all_examples

def print_communication_debug(puzzle_tensor, sender, receiver):
    """Print detailed debugging information about the communication between agents."""
    import torch.nn.functional as F
    
    print("\n  Communication Example:")
    print_grid(puzzle_tensor[0], "Original Puzzle")
    
    # Sender to Receiver communication
    print("\nSender → Receiver:")
    symbols1, symbol_logits1, _ = sender.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols1, "Sender")
    
    # Print position-specific symbol distributions
    print_positional_symbol_probabilities(symbol_logits1, sender, "Sender")
    
    reconstructed1, grid_logits1, intermediates1, confidences1, size_logits1 = receiver.decode_message_to_puzzle(
        symbols1, 
        temperature=0.1
    )
    
    # Print confidence scores
    print("\nDecoder Confidence Scores:")
    for i, conf in enumerate(confidences1):
        print(f"  Step {i+1}: {conf.item():.4f}")
    
    print_grid(reconstructed1.argmax(dim=-1)[0], "Sender→Receiver Reconstruction")

    # Receiver to Sender communication
    print("\nReceiver → Sender:")
    symbols2, symbol_logits2, _ = receiver.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols2, "Receiver")
    
    # Print position-specific symbol distributions
    print_positional_symbol_probabilities(symbol_logits2, receiver, "Receiver")
    
    reconstructed2, grid_logits2, intermediates2, confidences2, size_logits2 = sender.decode_message_to_puzzle(
        symbols2, 
        temperature=0.1
    )
    
    # Print confidence scores
    print("\nDecoder Confidence Scores:")
    for i, conf in enumerate(confidences2):
        print(f"  Step {i+1}: {conf.item():.4f}")
    
    print_grid(reconstructed2.argmax(dim=-1)[0], "Receiver→Sender Reconstruction")

def print_positional_symbol_probabilities(symbol_logits, agent, agent_name):
    """
    Print symbol probabilities broken down by position, showing which symbols are allowed.
    """
    import torch.nn.functional as F
    
    batch_size, seq_len, num_comm = symbol_logits.shape
    symbol_probs = F.softmax(symbol_logits[0], dim=-1)  # [seq_len, num_comm]
    
    print(f"\n{agent_name} Symbol Probabilities (Position-Specific):")
    
    # Get position mapping from agent
    position_to_symbols = getattr(agent, 'position_to_symbols', {})
    puzzle_symbols = getattr(agent, 'puzzle_symbols', 10)
    
    for pos in range(seq_len):
        print(f"  Position {pos}:")
        
        # Get allowed symbols for this position
        allowed_abs_symbols = position_to_symbols.get(pos, [])
        allowed_comm_symbols = [s - puzzle_symbols for s in allowed_abs_symbols if s >= puzzle_symbols]
        
        if allowed_comm_symbols:
            print(f"    Allowed comm symbols: {allowed_comm_symbols}")
            print(f"    Probabilities:")
            
            # Show probabilities for allowed symbols
            total_allowed_prob = 0.0
            for comm_idx in allowed_comm_symbols:
                if comm_idx < num_comm:
                    prob = symbol_probs[pos, comm_idx].item()
                    total_allowed_prob += prob
                    print(f"      Symbol {comm_idx}: {prob:.4f}")
            
            # Show probabilities for some disallowed symbols for comparison
            print(f"    Disallowed symbols (showing first 3):")
            disallowed_count = 0
            for comm_idx in range(min(num_comm, 10)):  # Check first 10 symbols max
                if comm_idx not in allowed_comm_symbols and disallowed_count < 3:
                    prob = symbol_probs[pos, comm_idx].item()
                    print(f"      Symbol {comm_idx}: {prob:.4f} (masked)")
                    disallowed_count += 1
            
            print(f"    Total probability on allowed symbols: {total_allowed_prob:.4f}")
        else:
            print(f"    No specific constraints (showing first 6):")
            for i in range(min(num_comm, 6)):
                prob = symbol_probs[pos, i].item()
                print(f"      Symbol {i}: {prob:.4f}")
        
        print()  # Empty line between positions
    
    # Also show overall symbol distribution across all positions
    print(f"  Overall Symbol Distribution (all positions combined):")
    overall_probs = symbol_probs.mean(dim=0)  # Average across positions
    for i in range(min(num_comm, 10)):  # Show first 10 symbols
        prob = overall_probs[i].item()
        print(f"    Comm Symbol {i}: {prob:.4f}")

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

def print_grid(grid: torch.Tensor, title: str = "Grid"):
    """Print a grid in a readable format"""
    print(f"\n{title}:")
    for row in grid.cpu().numpy():
        print("  " + " ".join(f"{x:2d}" for x in row))

def print_symbol_consistency(puzzle_to_symbols):
    """
    Print statistics about symbol consistency for each puzzle.
    This helps track whether encoders are developing consistent mappings.
    """
    print("\nSymbol Consistency Check:")
    for puzzle_id, symbols in puzzle_to_symbols.items():
        # Calculate most common symbol for each encoder
        encoder1_symbols = symbols['encoder1'][-20:]  # Look at recent history
        encoder2_symbols = symbols['encoder2'][-20:]
        
        if encoder1_symbols and encoder2_symbols:
            from collections import Counter
            encoder1_counter = Counter(encoder1_symbols)
            encoder2_counter = Counter(encoder2_symbols)
            
            encoder1_most_common = encoder1_counter.most_common(1)[0]
            encoder2_most_common = encoder2_counter.most_common(1)[0]
            
            encoder1_consistency = encoder1_most_common[1] / len(encoder1_symbols)
            encoder2_consistency = encoder2_most_common[1] / len(encoder2_symbols)
            
            print(f"  {puzzle_id}:")
            print(f"    Encoder1: Symbol {encoder1_most_common[0]} used {encoder1_consistency:.1%} of the time")
            print(f"    Encoder2: Symbol {encoder2_most_common[0]} used {encoder2_consistency:.1%} of the time")

def print_positional_constraints_status(dynamic_manager):
    """
    Print the current positional constraints status.
    """
    status = dynamic_manager.get_status()
    position_mapping = status.get('position_to_symbols', {})
    
    print("\nCurrent Positional Constraints:")
    print(f"  Sequence Length: {status['current_seq_length']}")
    print(f"  Total Symbols: {status['current_total_symbols']} (Comm: {status['current_comm_symbols']})")
    print("  Position → Allowed Symbols:")
    
    for position in sorted(position_mapping.keys()):
        symbols = position_mapping[position]
        comm_symbols = [s - status['puzzle_symbols'] for s in symbols if s >= status['puzzle_symbols']]
        print(f"    Position {position}: absolute symbols {symbols} → comm symbols {comm_symbols}")

def train_cycle(cycle, trainer, puzzles, device, puzzle_to_symbols, dynamic_manager):
    """Train for one cycle through all puzzles"""
    cycle_metrics = []
    
    # Visualization frequency (show detailed output every N cycles)
    visualization_frequency = 50
    
    for puzzle_idx, puzzle in enumerate(puzzles):
        puzzle_id = f"puzzle_{puzzle_idx}"
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
        
        # Show detailed communication debug for each puzzle every N cycles
        if cycle % visualization_frequency == 0:
            print(f"\n--- Visualization for Cycle {cycle}, Puzzle {puzzle_idx} ---")
            print_communication_debug(puzzle_tensor, trainer.agent1, trainer.agent2)
            
            # Show positional constraints status
            if puzzle_idx == 0:  # Only show once per cycle
                print_positional_constraints_status(dynamic_manager)
        
        # Track symbol usage for consistency checking
        if trainer.training_mode in ["encoder_only", "joint"]:
            for metrics in step_metrics:
                if 'encoder1_symbol' in metrics and 'encoder2_symbol' in metrics:
                    if puzzle_id not in puzzle_to_symbols:
                        puzzle_to_symbols[puzzle_id] = {
                            'encoder1': [],
                            'encoder2': []
                        }
                    
                    puzzle_to_symbols[puzzle_id]['encoder1'].append(metrics['encoder1_symbol'])
                    puzzle_to_symbols[puzzle_id]['encoder2'].append(metrics['encoder2_symbol'])
                    
                    # Print symbol consistency info periodically
                    if (cycle + 1) % 50 == 0 and puzzle_idx == 0:
                        print_symbol_consistency(puzzle_to_symbols)
    
    return cycle_metrics

def plot_training_progress(metrics_history, accuracies_history, symbol_history, config):
    """Create comprehensive training progress plots including sequence length evolution and positional mapping"""
    plt.figure(figsize=(15, 20))
    
    # Loss plot (similar to original train.py)
    plt.subplot(5, 1, 1)
    losses = [m['total_loss'] for m in metrics_history if not np.isnan(m['total_loss'])]
    plt.plot(losses, label='Loss', alpha=0.7)
    
    # Mark symbol additions with vertical lines
    for addition in symbol_history:
        cycle = addition['cycle']
        if cycle < len(losses):
            plt.axvline(x=cycle, color='red', linestyle='--', alpha=0.8, linewidth=2)
            plt.text(cycle, plt.ylim()[1]*0.9, f"+{addition['comm_symbols']}s,{addition['seq_length']}l", 
                    rotation=90, color='red', fontweight='bold', fontsize=10)
    
    plt.title('Training Loss with Positional Constraints')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Accuracy plot (like original train.py but with symbol markers)
    plt.subplot(5, 1, 2)
    plt.plot(accuracies_history['acc1_size'], label='Agent1 Size Acc', alpha=0.7)
    plt.plot(accuracies_history['acc1_content'], label='Agent1 Content Acc', alpha=0.7)
    plt.plot(accuracies_history['acc2_size'], label='Agent2 Size Acc', alpha=0.7)
    plt.plot(accuracies_history['acc2_content'], label='Agent2 Content Acc', alpha=0.7)
    
    # Mark symbol additions
    for addition in symbol_history:
        cycle = addition['cycle']
        if cycle < len(accuracies_history['acc1_content']):
            plt.axvline(x=cycle, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.title('Accuracies with Positional Symbol Constraints')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Symbol count over time (modified to show both symbols and sequence length)
    plt.subplot(5, 1, 3)
    symbol_counts = []
    current_symbols = config.symbols.initial_total_symbols
    
    for i in range(len(accuracies_history['acc1_content'])):
        for addition in symbol_history:
            if addition['cycle'] == i:
                current_symbols = addition['total_symbols']
        symbol_counts.append(current_symbols)
    
    plt.plot(symbol_counts, label='Total Symbols', linewidth=2, color='green')
    plt.plot([s - config.symbols.puzzle_symbols for s in symbol_counts], 
             label='Communication Symbols', linewidth=2, color='orange')
    
    plt.title('Symbol Count Evolution (Positional)')
    plt.xlabel('Step')
    plt.ylabel('Number of Symbols')
    plt.grid(True)
    plt.legend()
    
    # Sequence length evolution plot
    plt.subplot(5, 1, 4)
    seq_lengths = []
    current_seq_length = config.symbols.initial_seq_length
    
    for i in range(len(accuracies_history['acc1_content'])):
        for addition in symbol_history:
            if addition['cycle'] == i:
                current_seq_length = addition['seq_length']
        seq_lengths.append(current_seq_length)
    
    plt.plot(seq_lengths, label='Sequence Length', linewidth=2, color='purple')
    
    # Mark sequence length increases
    for addition in symbol_history:
        cycle = addition['cycle']
        if cycle < len(seq_lengths):
            plt.axvline(x=cycle, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.title('Sequence Length Evolution (Positional)')
    plt.xlabel('Step')
    plt.ylabel('Sequence Length')
    plt.grid(True)
    plt.legend()
    
    # NEW: Positional mapping evolution
    plt.subplot(5, 1, 5)
    positions_per_cycle = []
    
    for i in range(len(accuracies_history['acc1_content'])):
        max_position = 0
        for addition in symbol_history:
            if addition['cycle'] <= i and 'position_mapping' in addition:
                max_position = max(max_position, max(addition['position_mapping'].keys()) if addition['position_mapping'] else 0)
        positions_per_cycle.append(max_position + 1)  # +1 because positions are 0-indexed
    
    plt.plot(positions_per_cycle, label='Active Positions', linewidth=2, color='brown')
    
    # Mark position additions
    for addition in symbol_history:
        cycle = addition['cycle']
        if cycle < len(positions_per_cycle):
            plt.axvline(x=cycle, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.title('Active Positions Evolution')
    plt.xlabel('Step')
    plt.ylabel('Number of Active Positions')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics_positional.png', dpi=150, bbox_inches='tight')
    plt.close()

def log_cycle_metrics_dynamic(log_file, cycle, cycle_metrics, 
                            acc1_size_ma, acc1_content_ma, acc2_size_ma, acc2_content_ma,
                            current_comm_symbols, current_seq_length):
    """Enhanced logging that includes symbol and sequence length information"""
    # Calculate average metrics for this cycle
    avg_metrics = {
        'total_loss': np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])]),
        'training_mode': cycle_metrics[0]['training_mode'] if 'training_mode' in cycle_metrics[0] else 'unknown'
    }
    
    # Write to log file
    log_file.write(
        f"Cycle {cycle + 1}: " + 
        f"Loss={avg_metrics['total_loss']:.4f}, " +
        f"CommSyms={current_comm_symbols}, " +
        f"SeqLen={current_seq_length}, " +
        f"Size_Acc1={acc1_size_ma.get_average():.3f}, " +
        f"Content_Acc1={acc1_content_ma.get_average():.3f}, " +
        f"Size_Acc2={acc2_size_ma.get_average():.3f}, " +
        f"Content_Acc2={acc2_content_ma.get_average():.3f}\n"
    )
    log_file.flush()

def main():
    """Main training function with positional constraints"""
    
    # Print experiment configuration
    print("=" * 60)
    print("POSITIONAL SYMBOL & SEQUENCE LENGTH TRAINING")
    print("=" * 60)
    print(f"Plateau Detection:")
    print(f"  Cycles to check: {EXPERIMENT_CONFIG.plateau.plateau_cycles}")
    print(f"  Threshold: {EXPERIMENT_CONFIG.plateau.plateau_threshold:.1%}")
    print(f"  Min cycles before detection: {EXPERIMENT_CONFIG.plateau.min_cycles_before_detection}")
    print(f"\nPositional Symbol & Sequence Configuration:")
    print(f"  Starting symbols: {EXPERIMENT_CONFIG.symbols.initial_total_symbols} total ({EXPERIMENT_CONFIG.symbols.initial_comm_symbols} communication)")
    print(f"  Maximum symbols: {EXPERIMENT_CONFIG.symbols.max_total_symbols} total ({EXPERIMENT_CONFIG.symbols.max_comm_symbols} communication)")
    print(f"  Starting sequence length: {EXPERIMENT_CONFIG.symbols.initial_seq_length}")
    print(f"  Maximum sequence length: {EXPERIMENT_CONFIG.symbols.max_seq_length}")
    print(f"  Growth pattern: +2 symbols per position, +1 sequence length per plateau")
    print(f"  Position constraints: Position 0 → symbols 0-1, Position 1 → symbols 2-3, etc.")
    print(f"\nTraining:")
    print(f"  Total cycles: {EXPERIMENT_CONFIG.total_cycles}")
    print(f"  Learning rate: {EXPERIMENT_CONFIG.learning_rate}")
    print("=" * 60)
    
    # Save configuration for reference
    EXPERIMENT_CONFIG.save_to_json("current_experiment_config_positional.json")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=EXPERIMENT_CONFIG.symbols.initial_total_symbols,
        puzzle_symbols=EXPERIMENT_CONFIG.symbols.puzzle_symbols,
        max_seq_length=EXPERIMENT_CONFIG.symbols.initial_seq_length,  # Start with initial sequence length
        sender_scale=1.0
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=EXPERIMENT_CONFIG.symbols.initial_total_symbols,
        puzzle_symbols=EXPERIMENT_CONFIG.symbols.puzzle_symbols,
        max_seq_length=EXPERIMENT_CONFIG.symbols.initial_seq_length,  # Start with initial sequence length
        sender_scale=1.0 
    ).to(device)
    
    # Create dynamic training manager (now with positional constraints)
    dynamic_manager = DynamicTrainingManager(
        agents=[sender, receiver],
        plateau_cycles=EXPERIMENT_CONFIG.plateau.plateau_cycles,
        plateau_threshold=EXPERIMENT_CONFIG.plateau.plateau_threshold,
        min_cycles_before_detection=EXPERIMENT_CONFIG.plateau.min_cycles_before_detection,
        initial_total_symbols=EXPERIMENT_CONFIG.symbols.initial_total_symbols,
        puzzle_symbols=EXPERIMENT_CONFIG.symbols.puzzle_symbols,
        max_total_symbols=EXPERIMENT_CONFIG.symbols.max_total_symbols,
        initial_seq_length=EXPERIMENT_CONFIG.symbols.initial_seq_length,
        max_seq_length=EXPERIMENT_CONFIG.symbols.max_seq_length,
        device=device
    )
    
    # Create trainer
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=EXPERIMENT_CONFIG.learning_rate,
        device=device
    )
    trainer.set_training_mode("joint")
    
    # Load puzzles
    try:
        puzzles = load_arc_puzzles('arc-agi_test_challenges.json')
        # Use same selection as train.py for consistency
        selected_puzzles = [puzzles[1], test_puzzles[41]]
        # selected_puzzles = [puzzles[i] for i in range(0, len(puzzles), 10)][:10]
        print(f"Using {len(selected_puzzles)} puzzles for training (same as train.py)")
    except FileNotFoundError:
        print("ARC dataset not found. Please ensure 'arc-agi_test_challenges.json' is available.")
        return
    
    # Training setup
    metrics_history = []
    accuracies_history = {
        'acc1_content': [],
        'acc2_content': [],
        'acc1_size': [],
        'acc2_size': []
    }
    
    # Moving averages for plateau detection
    ma_window = 50
    acc1_content_ma = MovingAverage(ma_window)
    acc2_content_ma = MovingAverage(ma_window)
    acc1_size_ma = MovingAverage(ma_window)
    acc2_size_ma = MovingAverage(ma_window)
    
    # Training loop
    with open(EXPERIMENT_CONFIG.log_file, 'w') as log_file:
        # Write configuration to log
        log_file.write("POSITIONAL SYMBOL & SEQUENCE LENGTH TRAINING LOG\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Configuration: {EXPERIMENT_CONFIG.to_dict()}\n\n")
        
        puzzle_to_symbols = {}  # Track symbol usage like in original train.py
        
        for cycle in range(EXPERIMENT_CONFIG.total_cycles):
            # Show detailed cycle information like original train.py
            print(f"\nCycle {cycle + 1}/{EXPERIMENT_CONFIG.total_cycles}")
            
            # Get current symbol and sequence length status
            status = dynamic_manager.get_status()
            current_comm_symbols = status['current_comm_symbols']
            current_seq_length = status['current_seq_length']
            
            # Print status periodically
            if cycle % 50 == 0:
                print(f"  Current symbols: {status['current_total_symbols']} total, {current_comm_symbols} communication")
                print(f"  Current sequence length: {current_seq_length}")
                if status['addition_history']:
                    last_addition = status['addition_history'][-1]
                    cycles_since = cycle - last_addition['cycle']
                    print(f"  Last addition {cycles_since} cycles ago at cycle {last_addition['cycle']}")
                    print(f"    Added: {last_addition['comm_symbols']} comm symbols, seq length {last_addition['seq_length']}")
            
            # Train one cycle (with detailed output like original train.py)
            cycle_metrics = train_cycle(cycle, trainer, selected_puzzles, device, puzzle_to_symbols, dynamic_manager)
            metrics_history.extend(cycle_metrics)
            
            # Update moving averages
            for metrics in cycle_metrics:
                acc1_content_ma.update(metrics['agent1_content_accuracy'])
                acc2_content_ma.update(metrics['agent2_content_accuracy'])
                acc1_size_ma.update(metrics['agent1_size_accuracy'])
                acc2_size_ma.update(metrics['agent2_size_accuracy'])
            
            # Store history
            accuracies_history['acc1_content'].append(acc1_content_ma.get_average())
            accuracies_history['acc2_content'].append(acc2_content_ma.get_average())
            accuracies_history['acc1_size'].append(acc1_size_ma.get_average())
            accuracies_history['acc2_size'].append(acc2_size_ma.get_average())
            
            # Check for plateau and add symbols if needed
            avg_metrics = {
                'agent1_content_accuracy': acc1_content_ma.get_average(),
                'agent2_content_accuracy': acc2_content_ma.get_average(),
            }
            
            symbols_added = dynamic_manager.update_and_check(cycle, avg_metrics)
            
            if symbols_added:
                status = dynamic_manager.get_status()
                addition = status['addition_history'][-1]
                print(f"\n🎯 Positional symbols added at cycle {cycle}! Total: {addition['total_symbols']} (Comm: {addition['comm_symbols']}, SeqLen: {addition['seq_length']})")
                
                # Log symbol addition
                log_file.write(f"POSITIONAL SYMBOLS ADDED - Cycle {cycle}\n")
                log_file.write(f"  Total symbols: {addition['total_symbols']}\n")
                log_file.write(f"  Communication symbols: {addition['comm_symbols']}\n")
                log_file.write(f"  Sequence length: {addition['seq_length']}\n")
                log_file.write(f"  Accuracy before: Agent1={avg_metrics['agent1_content_accuracy']:.3f}, Agent2={avg_metrics['agent2_content_accuracy']:.3f}\n")
                if 'position_mapping' in addition:
                    log_file.write(f"  Position mapping: {addition['position_mapping']}\n")
                log_file.write("\n")
                
                # Recreate optimizers to include new parameters
                trainer.opt1 = torch.optim.Adam([
                    {'params': sender.embedding_system.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': sender.encoder.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': sender.decoder.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': sender.communication_embedding.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate}
                ])
                
                trainer.opt2 = torch.optim.Adam([
                    {'params': receiver.embedding_system.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': receiver.encoder.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': receiver.decoder.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate},
                    {'params': receiver.communication_embedding.parameters(), 'lr': EXPERIMENT_CONFIG.learning_rate}
                ])
            
            # Log metrics (like original train.py)
            log_cycle_metrics_dynamic(log_file, cycle, cycle_metrics, 
                                   acc1_size_ma, acc1_content_ma, acc2_size_ma, acc2_content_ma,
                                   current_comm_symbols, current_seq_length)
            
            # Plot metrics more frequently (like original train.py)
            if (cycle + 1) % 10 == 0:
                plot_training_progress(
                    metrics_history, 
                    accuracies_history,
                    dynamic_manager.symbol_manager.symbol_addition_history,
                    EXPERIMENT_CONFIG
                )
    
    # Final summary
    final_status = dynamic_manager.get_status()
    print("\n" + "=" * 60)
    print("POSITIONAL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Cycles completed: {EXPERIMENT_CONFIG.total_cycles}")
    print(f"Final symbol count: {final_status['current_total_symbols']} total ({final_status['current_comm_symbols']} communication)")
    print(f"Final sequence length: {final_status['current_seq_length']}")
    print(f"Additions made: {len(final_status['addition_history'])}")
    
    if final_status['addition_history']:
        print(f"\nPositional symbol & sequence length progression:")
        print(f"  Start: {EXPERIMENT_CONFIG.symbols.initial_comm_symbols} comm symbols, seq length {EXPERIMENT_CONFIG.symbols.initial_seq_length}")
        print(f"    Position 0: symbols 0-1")
        
        for addition in final_status['addition_history']:
            print(f"  Cycle {addition['cycle']:4d}: -> {addition['comm_symbols']} comm symbols, seq length {addition['seq_length']}")
            if 'position_mapping' in addition:
                new_positions = [pos for pos in addition['position_mapping'].keys() if pos > 0]
                for pos in new_positions:
                    symbols = addition['position_mapping'][pos]
                    comm_symbols = [s - EXPERIMENT_CONFIG.symbols.puzzle_symbols for s in symbols]
                    print(f"    Position {pos}: symbols {symbols} (comm: {comm_symbols})")
    
    print(f"\nFinal accuracy: Agent1={acc1_content_ma.get_average():.3f}, Agent2={acc2_content_ma.get_average():.3f}")
    print(f"Results saved to: {EXPERIMENT_CONFIG.log_file}")
    print("Plot saved to: training_metrics_positional.png")

if __name__ == "__main__":
    main()