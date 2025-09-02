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

def pretrain_cross_agent_communication_selection(agent1, agent2, puzzles, device, 
                    encoder_epochs=None,   
                    decoder_epochs=None,   # Not used in selection task but kept for compatibility
                    learning_rate=0.001, 
                    diversity_weight=0.1,
                    regularization_weight=0.001,
                    visualization_frequency=50,
                    num_distractors=3):
    """
    Simplified pre-training for selection task: Train Agent1's encoder, then copy to Agent2.
    Focus on encoder training since we don't need decoder reconstruction.
    """
    import torch.nn.functional as F
    
    # Handle epoch parameters
    if encoder_epochs is None:
        encoder_epochs = 200  # Default for selection task
    
    print(f"\n===== Starting Selection Task Pre-training =====")
    
    num_comm_symbols = agent1.current_comm_symbols
    print(f"Using {num_comm_symbols} communication symbols for pretraining")
    print(f"Selection task with {num_distractors} distractors per target")
    
    # Convert puzzles to tensors
    puzzle_tensors = [
        torch.tensor(puzzle.test_input, dtype=torch.long, device=device).unsqueeze(0)
        for puzzle in puzzles[:20]  # Limit for efficiency
    ]
    
    # Assign target symbols to puzzles for encoder training
    targets = {}
    for i, _ in enumerate(puzzle_tensors):
        targets[i] = i % min(num_comm_symbols, len(puzzle_tensors))
    
    print(f"Assigned {len(set(targets.values()))} unique target symbols to {len(puzzle_tensors)} puzzles")
    
    # Training history
    history = {
        'encoder_loss': [],
        'encoder_accuracy': [],
        'selection_accuracy': [],  # NEW: Track selection performance
        'phase': []
    }
    
    print(f"\nSelection Pre-training Schedule:")
    print(f"Phase 1: Training Agent1 encoder for {encoder_epochs} epochs")
    print(f"Phase 2: Copy Agent1 weights to Agent2")
    print(f"Phase 3: Test cross-agent selection")
    
    # =================================================================
    # PHASE 1: TRAIN Agent1 ENCODER ONLY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Agent1 ENCODER TRAINING ({encoder_epochs} epochs)")
    print(f"{'='*60}")
    
    # Set up encoder-only training for Agent1
    encoder_params = []
    for name, param in agent1.named_parameters():
        if any(component in name for component in ['encoder', 'embedding_system', 'message_pooling']):
            encoder_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Disable Agent2 entirely during this phase
    for param in agent2.parameters():
        param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=learning_rate)
    
    for epoch in range(encoder_epochs):
        total_loss = 0.0
        correct = 0
        
        indices = torch.randperm(len(puzzle_tensors))
        
        show_visualization = (epoch % visualization_frequency == 0) or (epoch == encoder_epochs - 1)
        if show_visualization:
            print(f"\n--- Phase 1 Encoder Visualization (Epoch {epoch+1}) ---")
        
        for idx in indices:
            i = idx.item()
            puzzle_tensor = puzzle_tensors[i]
            
            encoder_optimizer.zero_grad()
            
            # Forward pass through Agent1's encoder
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1
            )
            
            pred_symbol = symbols[0, 0].argmax().item()
            target = torch.tensor([targets[i]], device=device)
            
            # Symbol prediction loss
            symbol_loss = F.cross_entropy(symbol_logits[0, 0].unsqueeze(0), target)
            
            # Regularization
            reg_loss = 0.0
            for param in encoder_params:
                reg_loss += param.pow(2.0).sum()
            
            total_loss_item = symbol_loss + regularization_weight * reg_loss
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            encoder_optimizer.step()
            
            # Track metrics
            if pred_symbol == targets[i]:
                correct += 1
            total_loss += total_loss_item.item()
            
            # Show encoder learning during visualization
            if show_visualization and i < 3:
                print(f"\nPuzzle {i} (Encoder Learning):")
                print(f"  Target symbol: {targets[i]}, Predicted: {pred_symbol} {'✓' if pred_symbol == targets[i] else '✗'}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(puzzle_tensors)
        accuracy = correct / len(puzzle_tensors)
        
        history['encoder_loss'].append(avg_loss)
        history['encoder_accuracy'].append(accuracy)
        history['selection_accuracy'].append(0.0)  # Placeholder
        history['phase'].append('encoder')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Encoder Epoch {epoch+1}/{encoder_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
    
    print(f"\nPhase 1 complete. Final encoder accuracy: {accuracy:.3f}")

    # =================================================================
    # PHASE 2: COPY Agent1's weights to Agent2
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: COPYING Agent1 WEIGHTS TO Agent2")
    print(f"{'='*60}")
    
    with torch.no_grad():
        total_copied_params = 0
        
        # Copy encoder weights
        print("Copying encoder weights...")
        encoder_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.encoder.named_parameters(), agent2.encoder.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                encoder_params_copied += param1.numel()
        
        print(f"  ✓ Encoder weights copied ({encoder_params_copied:,} parameters)")
        total_copied_params += encoder_params_copied
        
        # Copy embedding system weights
        print("Copying embedding system weights...")
        embedding_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.embedding_system.named_parameters(), agent2.embedding_system.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                embedding_params_copied += param1.numel()
        
        print(f"  ✓ Embedding system weights copied ({embedding_params_copied:,} parameters)")
        total_copied_params += embedding_params_copied
        
        # Copy message pooling weights
        print("Copying message pooling weights...")
        pooling_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.message_pooling.named_parameters(), agent2.message_pooling.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                pooling_params_copied += param1.numel()
        
        print(f"  ✓ Message pooling weights copied ({pooling_params_copied:,} parameters)")
        total_copied_params += pooling_params_copied
        
        # Copy communication embeddings
        start_idx = agent1.puzzle_symbols
        end_idx = start_idx + agent1.current_comm_symbols
        
        agent2.communication_embedding.weight[start_idx:end_idx].copy_(
            agent1.communication_embedding.weight[start_idx:end_idx]
        )
        
        comm_params_copied = agent1.current_comm_symbols * agent1.communication_embedding.embedding_dim
        print(f"  ✓ Communication embeddings copied ({comm_params_copied:,} parameters)")
        total_copied_params += comm_params_copied
        
        # Sync vocabulary states
        agent2.current_comm_symbols = agent1.current_comm_symbols
        agent2.current_seq_length = agent1.current_seq_length
        agent2.current_total_symbols = agent1.current_total_symbols
        agent2.communication_vocabulary = agent1.communication_vocabulary.copy()
        
        print(f"\nWeight copying complete! Total parameters: {total_copied_params:,}")
        print(f"{'='*60}")
    
    # =================================================================
    # PHASE 3: TEST CROSS-AGENT SELECTION
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: CROSS-AGENT SELECTION EVALUATION")
    print(f"{'='*60}")
    
    # Re-enable all parameters
    for param in agent1.parameters():
        param.requires_grad = True
    for param in agent2.parameters():
        param.requires_grad = True
    
    agent1_to_agent2_correct = 0
    agent2_to_agent1_correct = 0
    
    print(f"\nTesting cross-agent selection with {num_distractors} distractors:")
    for i, puzzle in enumerate(puzzle_tensors):
        if i >= 10:  # Limit output
            break
            
        with torch.no_grad():
            # Test Agent1 → Agent2 communication
            symbols1, _, _ = agent1.encode_puzzle_to_message(puzzle, temperature=0.1)
            
            # Create candidates: target + random distractors
            candidates = [puzzle]
            available_puzzles = [p for j, p in enumerate(puzzle_tensors) if j != i]
            distractor_indices = np.random.choice(len(available_puzzles), 
                                                min(num_distractors, len(available_puzzles)), 
                                                replace=False)
            for idx in distractor_indices:
                candidates.append(available_puzzles[idx])
            
            # Agent2 selects
            selection_probs1, selection_logits1, _ = agent2.select_from_candidates(
                symbols1, candidates, temperature=0.1
            )
            
            pred1 = selection_logits1.argmax(dim=-1).item()
            if pred1 == 0:  # Target is at index 0
                agent1_to_agent2_correct += 1
            
            # Test Agent2 → Agent1 communication (reverse)
            symbols2, _, _ = agent2.encode_puzzle_to_message(puzzle, temperature=0.1)
            
            # Create different set of candidates
            candidates2 = [puzzle]
            distractor_indices2 = np.random.choice(len(available_puzzles), 
                                                 min(num_distractors, len(available_puzzles)), 
                                                 replace=False)
            for idx in distractor_indices2:
                candidates2.append(available_puzzles[idx])
            
            selection_probs2, selection_logits2, _ = agent1.select_from_candidates(
                symbols2, candidates2, temperature=0.1
            )
            
            pred2 = selection_logits2.argmax(dim=-1).item()
            if pred2 == 0:
                agent2_to_agent1_correct += 1
            
            print(f"Puzzle {i}:")
            print(f"  Agent1→Agent2: Selected {pred1} {'✓' if pred1 == 0 else '✗'} (conf: {selection_probs1[0, 0]:.3f})")
            print(f"  Agent2→Agent1: Selected {pred2} {'✓' if pred2 == 0 else '✗'} (conf: {selection_probs2[0, 0]:.3f})")
    
    test_count = min(10, len(puzzle_tensors))
    
    print(f"\nFinal Cross-Agent Selection Results:")
    print(f"  Agent1→Agent2: {agent1_to_agent2_correct}/{test_count} = {agent1_to_agent2_correct/test_count:.3f}")
    print(f"  Agent2→Agent1: {agent2_to_agent1_correct}/{test_count} = {agent2_to_agent1_correct/test_count:.3f}")
    
    # Plot results
    plot_selection_pretraining_results(history, encoder_epochs)
    
    return history

def plot_selection_pretraining_results(history, encoder_epochs):
    """Plot results from selection pre-training"""
    plt.figure(figsize=(15, 10))
    
    epochs = len(history['encoder_loss'])
    epoch_nums = list(range(1, epochs + 1))
    
    # Plot encoder loss
    plt.subplot(2, 2, 1)
    encoder_losses = [loss for loss, phase in zip(history['encoder_loss'], history['phase']) if phase == 'encoder']
    encoder_x = list(range(1, len(encoder_losses) + 1))
    
    plt.plot(encoder_x, encoder_losses, label='Encoder Loss', linewidth=2, color='blue')
    plt.title('Selection Task: Encoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot encoder accuracy
    plt.subplot(2, 2, 2)
    encoder_accs = [acc for acc, phase in zip(history['encoder_accuracy'], history['phase']) if phase == 'encoder']
    
    plt.plot(encoder_x, encoder_accs, label='Encoder Accuracy', linewidth=2, color='blue')
    plt.title('Selection Task: Symbol Assignment Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    
    # Task comparison
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.8, 'Selection Task vs Reconstruction', fontsize=14, fontweight='bold', ha='center')
    plt.text(0.1, 0.6, 'Selection Task Benefits:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.55, '• No pixel-perfect reconstruction needed', fontsize=10)
    plt.text(0.1, 0.5, '• More natural communication task', fontsize=10)
    plt.text(0.1, 0.45, '• Easier to evaluate and interpret', fontsize=10)
    plt.text(0.1, 0.4, '• Adjustable difficulty via distractors', fontsize=10)
    
    plt.text(0.1, 0.3, 'Architecture Changes:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.25, '• Symmetric encoding (both agents encode)', fontsize=10)
    plt.text(0.1, 0.2, '• Similarity-based selection', fontsize=10)
    plt.text(0.1, 0.15, '• Classification loss instead of reconstruction', fontsize=10)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Training Summary:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Encoder Epochs: {encoder_epochs}', fontsize=12)
    
    if encoder_accs:
        plt.text(0.1, 0.6, f'Final Encoder Acc: {encoder_accs[-1]:.3f}', fontsize=12)
    
    plt.text(0.1, 0.4, f'Task: Puzzle Selection', fontsize=12)
    plt.text(0.1, 0.3, f'Loss: Cross-Entropy Classification', fontsize=12)
    
    plt.text(0.1, 0.1, f'Ready for joint training!', fontsize=12, style='italic', color='green')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('selection_pretraining.png', dpi=150, bbox_inches='tight')
    plt.close()

# Updated plotting function for selection metrics
def plot_training_metrics_selection(metrics_history, accuracies_history, vocab_history=None, title="Selection Training Metrics"):
    """
    Plot training metrics for selection task (updated for selection accuracies)
    """
    plt.figure(figsize=(20, 15) if vocab_history else (15, 12))
    
    num_plots = 4 if vocab_history else 3
    
    # Plot loss
    plt.subplot(num_plots, 1, 1)
    losses = [m['total_loss'] for m in metrics_history if not np.isnan(m['total_loss'])]
    plt.plot(losses, label='Total Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot selection accuracies
    plt.subplot(num_plots, 1, 2)
    plt.plot(accuracies_history['acc1_selection'], label='Agent1 Selection Acc', alpha=0.8, linewidth=2)
    plt.plot(accuracies_history['acc2_selection'], label='Agent2 Selection Acc', alpha=0.8, linewidth=2)
    plt.title(f'{title} - Selection Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot confidence in correct selection
    plt.subplot(num_plots, 1, 3)
    plt.plot(accuracies_history['conf1_correct'], label='Agent1 Correct Confidence', alpha=0.7)
    plt.plot(accuracies_history['conf2_correct'], label='Agent2 Correct Confidence', alpha=0.7)
    plt.title(f'{title} - Confidence in Correct Selection')
    plt.xlabel('Step')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot vocabulary progression if available
    if vocab_history:
        plt.subplot(num_plots, 1, 4)
        plt.plot(vocab_history['cycles'], vocab_history['vocab_sizes'], 'o-', 
                label='Vocabulary Size', linewidth=2, markersize=4)
        plt.plot(vocab_history['cycles'], vocab_history['seq_lengths'], 's-', 
                label='Sequence Length', linewidth=2, markersize=4)
        
        # Add expansion markers
        for cycle in vocab_history.get('expansion_cycles', []):
            plt.axvline(x=cycle, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'{title} - Vocabulary Progression')
        plt.xlabel('Cycle')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('selection_training_metrics.png')
    plt.close()

def print_selection_debug(puzzle_tensor, sender, receiver, puzzle_dataset, puzzle_idx):
    """
    Debug function for selection task - shows how agents encode and select
    """
    print("\n  Selection Task Debug:")
    print_grid(puzzle_tensor[0], "Target Puzzle")
    
    # Show vocabulary status
    print(f"\nCurrent Vocabulary Status:")
    print(f"  Sender: {sender.current_comm_symbols} symbols, {sender.current_seq_length} length")
    print(f"  Receiver: {receiver.current_comm_symbols} symbols, {receiver.current_seq_length} length")
    print(f"  Similarity metric: {sender.similarity_metric}")
    
    # Sender encodes message
    print("\nSender → Receiver Selection:")
    symbols, symbol_logits, stats = sender.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols, "Sender")
    
    # Create selection candidates
    num_distractors = 3
    candidates = [puzzle_tensor]
    
    # Sample distractors
    available_indices = [i for i in range(len(puzzle_dataset)) if i != puzzle_idx]
    if len(available_indices) >= num_distractors:
        distractor_indices = np.random.choice(available_indices, num_distractors, replace=False)
        for idx in distractor_indices:
            distractor = torch.tensor(
                puzzle_dataset[idx].test_input, 
                dtype=torch.long, 
                device=puzzle_tensor.device
            ).unsqueeze(0)
            candidates.append(distractor)
    
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
    
    # NEW: Show incorrectly selected puzzle if there was a mistake
    if predicted_idx != 0:
        print(f"\n  ✗ INCORRECT SELECTION - Agent chose distractor {predicted_idx}:")
        incorrect_puzzle = candidates[predicted_idx]
        print_grid(incorrect_puzzle[0], f"Incorrectly Selected Puzzle (Distractor {predicted_idx})")
        
        # Show confidence comparison
        incorrect_confidence = selection_probs[0, predicted_idx].item()
        print(f"  Target confidence: {target_confidence:.4f}")
        print(f"  Incorrect choice confidence: {incorrect_confidence:.4f}")
        print(f"  Confidence difference: {incorrect_confidence - target_confidence:.4f}")
    
    print(f"\nAll selection probabilities:")
    for i, prob in enumerate(selection_probs[0]):
        marker = " ← target" if i == 0 else f" ← distractor {i}"
        symbol = "✓" if i == predicted_idx else " "
        print(f"    {symbol} Candidate {i}: {prob.item():.4f}{marker}")
    
    # Show similarity scores
    print(f"\nSimilarity scores (logits):")
    for i, score in enumerate(selection_logits[0]):
        marker = " ← target" if i == 0 else f" ← distractor {i}"
        symbol = "✓" if i == predicted_idx else " "
        print(f"    {symbol} Candidate {i}: {score.item():.4f}{marker}")

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
    
    # SELECTION MODIFICATION: Create selection agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,      # Maximum symbols
        puzzle_symbols=5,
        max_seq_length=1,    # Maximum sequence length
        sender_scale=1.0,
        similarity_metric='cosine'  # Options: 'cosine', 'dot', 'learned'
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,      # Maximum symbols
        puzzle_symbols=5,
        max_seq_length=1,    # Maximum sequence length
        sender_scale=1.0,
        similarity_metric='cosine'  # Should match sender
    ).to(device)
    
    # SELECTION MODIFICATION: Create selection trainer
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=1e-7,
        device=device,
        expansion_frequency=100000,      # Expand every 200 cycles
        symbols_per_expansion=5,      # Add 2 symbols per expansion
        length_per_expansion=1,       # Add 1 to sequence length per expansion
        sync_frequency=50,            # Synchronize every 50 cycles
        num_distractors=4,           # NEW: Number of distractor puzzles
        distractor_strategy='random'  # NEW: Distractor sampling strategy
    )
    
    # Load ARC puzzles
    arc_file_path = 'arc-agi_test_challenges.json'
    test_puzzles = load_arc_puzzles(arc_file_path)
    print(f"\nLoaded {len(test_puzzles)} total examples from ARC dataset")
    
    # Select puzzles for training (need enough for distractors)
    min_puzzles_needed = trainer.num_distractors + 5  # Need enough for distractors + some buffer
    if len(test_puzzles) < min_puzzles_needed:
        raise ValueError(f"Need at least {min_puzzles_needed} puzzles for selection task with {trainer.num_distractors} distractors")
    
    # selected_puzzles = [test_puzzles[0], test_puzzles[47]]  # Start with 2 puzzles
    selected_puzzles = [test_puzzles[i] for i in range(0, min(50, len(test_puzzles)), 7)][:5]  # More puzzles if needed
    print(f"Selected {len(selected_puzzles)} puzzles for training")
    
    # SELECTION MODIFICATION: Set puzzle dataset for distractor sampling
    trainer.set_puzzle_dataset(selected_puzzles)
    
    # Show initial vocabulary state
    print("\n" + "="*60)
    print("INITIAL VOCABULARY STATE")
    print("="*60)
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)

    # Optional pre-training phase for selection task
    encoder_pretrain_epochs = 0   # Set to > 0 to enable pre-training

    if encoder_pretrain_epochs > 0:
        selection_history = pretrain_cross_agent_communication_selection(
            sender, receiver, selected_puzzles, device, 
            encoder_epochs=encoder_pretrain_epochs,
            learning_rate=0.001,
            diversity_weight=0.2,
            regularization_weight=0.001,
            num_distractors=trainer.num_distractors
        )
        
        print(f"Selection pre-training completed:")
        print(f"  Encoder trained for {encoder_pretrain_epochs} epochs")
    
    # Main progressive selection training
    total_cycles = 1000
    
    print(f"\n--- Starting Progressive Selection Training ({total_cycles} cycles) ---")
    print(f"Selection task: {trainer.num_distractors + 1} candidates per trial")
    print(f"Distractor strategy: {trainer.distractor_strategy}")
    trainer.set_training_mode("joint")
    
    # Initialize histories and trackers for selection metrics
    metrics_history = []
    acc1_selection_history = []
    acc2_selection_history = []
    conf1_correct_history = []
    conf2_correct_history = []
    
    # Vocabulary progression tracking
    vocab_history = {
        'cycles': [],
        'vocab_sizes': [],
        'seq_lengths': [],
        'expansion_cycles': []
    }
    
    # Initialize moving averages
    ma_window = 50
    acc1_selection_ma = MovingAverage(ma_window)
    acc2_selection_ma = MovingAverage(ma_window)
    conf1_correct_ma = MovingAverage(ma_window)
    conf2_correct_ma = MovingAverage(ma_window)
    
    # Enhanced logging
    with open('selection_training_log.txt', 'w') as log_file:
        log_file.write("Progressive Selection Training Log\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Task: Puzzle Selection ({trainer.num_distractors + 1} candidates)\n")
        log_file.write(f"Distractor strategy: {trainer.distractor_strategy}\n")
        log_file.write(f"Similarity metric: {sender.similarity_metric}\n")
        log_file.write(f"Expansion every {trainer.expansion_frequency} cycles\n")
        log_file.write("="*50 + "\n\n")
        
        for cycle in range(total_cycles):
            print(f"\nCycle {cycle + 1}/{total_cycles}")
            
            # Track vocabulary progression
            if cycle == 0 or cycle % 20 == 0:
                vocab_history['cycles'].append(cycle + 1)
                vocab_history['vocab_sizes'].append(sender.current_comm_symbols)
                vocab_history['seq_lengths'].append(sender.current_seq_length)
            
            # Check for expansion events
            if cycle > 0 and cycle % trainer.expansion_frequency == 0:
                vocab_history['expansion_cycles'].append(cycle + 1)
                log_file.write(f"VOCABULARY EXPANSION at cycle {cycle + 1}\n")
                log_file.write(f"  New vocab size: {sender.current_comm_symbols}\n")
                log_file.write(f"  New seq length: {sender.current_seq_length}\n")
                log_file.write("-" * 40 + "\n")
            
            # Train on each puzzle with its index for distractor sampling
            cycle_metrics = []
            for puzzle_idx, puzzle in enumerate(selected_puzzles):
                puzzle_tensor = torch.tensor(
                    puzzle.test_input, 
                    dtype=torch.long, 
                    device=device
                ).unsqueeze(0)
                
                step_metrics = trainer.train_bidirectional_step(
                    puzzle_tensor, 
                    puzzle_idx,  # NEW: Pass puzzle index for distractor sampling
                    num_exchanges=1,
                    temperature=1.0,
                    initial_phase=False
                )
                
                cycle_metrics.extend(step_metrics)
                
                # Show debug info periodically
                if cycle % 10 == 0:
                    print(f"\n--- Visualization for Cycle {cycle}, Puzzle {puzzle_idx} ---")
                    print_selection_debug(puzzle_tensor, sender, receiver, selected_puzzles, puzzle_idx)
            
            # Update metrics and moving averages
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
            
            # Log cycle metrics
            if cycle_metrics:
                avg_metrics = {
                    'total_loss': np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])]),
                    'selection_acc1': np.mean([m['agent1_selection_accuracy'] for m in cycle_metrics]),
                    'selection_acc2': np.mean([m['agent2_selection_accuracy'] for m in cycle_metrics]),
                    'conf1': np.mean([m['agent1_correct_confidence'] for m in cycle_metrics]),
                    'conf2': np.mean([m['agent2_correct_confidence'] for m in cycle_metrics]),
                }
                
                log_file.write(
                    f"Cycle {cycle + 1}: " + 
                    f"Loss={avg_metrics['total_loss']:.4f}, " +
                    f"Vocab={sender.current_comm_symbols}, " +
                    f"Sel_Acc1={avg_metrics['selection_acc1']:.3f}, " +
                    f"Sel_Acc2={avg_metrics['selection_acc2']:.3f}, " +
                    f"Conf1={avg_metrics['conf1']:.3f}, " +
                    f"Conf2={avg_metrics['conf2']:.3f}\n"
                )
                log_file.flush()
            
            # Plot metrics periodically
            if (cycle + 1) % 10 == 0:
                accuracies_history = {
                    'acc1_selection': acc1_selection_history,
                    'acc2_selection': acc2_selection_history,
                    'conf1_correct': conf1_correct_history,
                    'conf2_correct': conf2_correct_history
                }
                plot_training_metrics_selection(metrics_history, accuracies_history, vocab_history, 
                                              title=f"Progressive Selection Training (Cycle {cycle+1})")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SELECTION TRAINING SUMMARY")
    print("="*60)

    if metrics_history:
        recent_metrics = metrics_history[-50:]  # Last 50 steps
        
        final_sel_acc1 = acc1_selection_ma.get_average()
        final_sel_acc2 = acc2_selection_ma.get_average()
        final_conf1 = conf1_correct_ma.get_average()
        final_conf2 = conf2_correct_ma.get_average()
        
        print(f"\nFinal Selection Performance:")
        print(f"  Agent 1 - Selection Accuracy: {final_sel_acc1:.3f}, Confidence: {final_conf1:.3f}")
        print(f"  Agent 2 - Selection Accuracy: {final_sel_acc2:.3f}, Confidence: {final_conf2:.3f}")
        
        print(f"\nTask Configuration:")
        print(f"  Candidates per trial: {trainer.num_distractors + 1}")
        print(f"  Distractor strategy: {trainer.distractor_strategy}")
        print(f"  Similarity metric: {sender.similarity_metric}")

    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)
    
    print("\nProgressive selection training complete! Check selection_training_log.txt for details")

if __name__ == "__main__":
    main()