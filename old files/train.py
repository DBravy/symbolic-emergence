import torch
# PROGRESSIVE MODIFICATION: Replace these imports
# from agent import Agent
# from trainer import CommunicationTrainer
from agent import ProgressiveAgent as Agent
from trainer import ProgressiveCommunicationTrainer as CommunicationTrainer

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

# Existing functions from train.py
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

# NEW FUNCTION: Supervised encoder pre-training
def pretrain_encoder(agent, puzzles, device, 
                    epochs=300, 
                    learning_rate=0.001, 
                    diversity_weight=0.1,
                    regularization_weight=0.001,
                    reconstruction_weight=1.0,
                    symbol_weight=1.0):
    """
    Separate pretraining of encoder and decoder to ensure consistent mappings.
    
    Phase 1: Train ONLY encoder (puzzle → symbol)
    Phase 2: Train ONLY decoder (target symbol → puzzle)
    
    This ensures both learn the same mapping:
    - Encoder learns: Puzzle 1 → Symbol 1
    - Decoder learns: Symbol 1 → Puzzle 1
    """
    print(f"\n===== Starting Separate Encoder/Decoder Pre-training for {agent.agent_id} =====")
    
    num_comm_symbols = agent.current_comm_symbols
    print(f"Using {num_comm_symbols} communication symbols for pretraining")
    print(f"Current vocabulary: puzzle_symbols={agent.puzzle_symbols}, comm_symbols={agent.current_comm_symbols}")
    
    # Convert puzzles to tensors
    puzzle_tensors = [
        torch.tensor(puzzle.test_input, dtype=torch.long, device=device).unsqueeze(0)
        for puzzle in puzzles[:20]  # Limit to first 20 puzzles for efficiency
    ]
    
    # Assign target symbols to puzzles
    targets = {}
    for i, _ in enumerate(puzzle_tensors):
        targets[i] = i % min(num_comm_symbols, len(puzzle_tensors))
    
    print(f"Assigned {len(set(targets.values()))} unique target symbols to {len(puzzle_tensors)} puzzles")
    print("Target mappings:")
    for i in range(min(5, len(puzzle_tensors))):
        print(f"  Puzzle {i} → Symbol {targets[i]}")
    
    # Training history
    history = {
        'encoder_loss': [],
        'decoder_loss': [],
        'encoder_accuracy': [],
        'decoder_accuracy': [],
        'unique_symbols': [],
        'phase': []
    }
    
    # Split epochs between encoder and decoder training
    encoder_epochs = epochs // 2
    decoder_epochs = epochs - encoder_epochs
    
    print(f"\nPhase 1: Training encoder for {encoder_epochs} epochs")
    print(f"Phase 2: Training decoder for {decoder_epochs} epochs")
    
    # =================================================================
    # PHASE 1: TRAIN ONLY ENCODER (puzzle → target symbol)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: ENCODER TRAINING")
    print(f"{'='*60}")
    
    # Set up encoder-only training
    encoder_params = []
    for name, param in agent.named_parameters():
        if any(component in name for component in ['encoder', 'embedding_system']):
            encoder_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=learning_rate)
    
    for epoch in range(encoder_epochs):
        total_loss = 0.0
        correct = 0
        
        indices = torch.randperm(len(puzzle_tensors))
        
        for idx in indices:
            i = idx.item()
            puzzle_tensor = puzzle_tensors[i]
            
            encoder_optimizer.zero_grad()
            
            # Forward pass through encoder only
            symbols, symbol_logits, _ = agent.encode_puzzle_to_message(
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
            
            # Diversity loss
            diversity_loss = 0.0
            if len(puzzle_tensors) > 1:
                current_probs = F.softmax(symbol_logits[0, 0], dim=-1)
                
                num_compared = 0
                other_indices = [j for j in range(len(puzzle_tensors)) if j != i]
                if len(other_indices) > 5:
                    other_indices = np.random.choice(other_indices, 5, replace=False)
                
                for j in other_indices:
                    if targets[j] != targets[i]:
                        with torch.no_grad():
                            _, other_logits, _ = agent.encode_puzzle_to_message(
                                puzzle_tensors[j], temperature=0.1
                            )
                            other_probs = F.softmax(other_logits[0, 0], dim=-1)
                            
                            similarity = F.cosine_similarity(
                                current_probs.unsqueeze(0), other_probs.unsqueeze(0)
                            )
                            diversity_loss += similarity
                            num_compared += 1
                
                if num_compared > 0:
                    diversity_loss = diversity_loss / num_compared
            
            total_loss_item = (
                symbol_loss +
                regularization_weight * reg_loss +
                diversity_weight * diversity_loss
            )
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            encoder_optimizer.step()
            
            # Track metrics
            if pred_symbol == targets[i]:
                correct += 1
            total_loss += total_loss_item.item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(puzzle_tensors)
        accuracy = correct / len(puzzle_tensors)
        
        # Check unique symbols
        unique_symbols = set()
        for puzzle in puzzle_tensors:
            with torch.no_grad():
                symbols, _, _ = agent.encode_puzzle_to_message(puzzle, temperature=0.1)
                symbol = symbols[0, 0].argmax().item()
                unique_symbols.add(symbol)
        
        history['encoder_loss'].append(avg_loss)
        history['encoder_accuracy'].append(accuracy)
        history['decoder_loss'].append(0.0)  # Placeholder
        history['decoder_accuracy'].append(0.0)  # Placeholder
        history['unique_symbols'].append(len(unique_symbols))
        history['phase'].append('encoder')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Encoder Epoch {epoch+1}/{encoder_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}, Unique={len(unique_symbols)}")
    
    print(f"\nEncoder training complete. Final accuracy: {accuracy:.3f}")
    
    # =================================================================
    # PHASE 2: TRAIN ONLY DECODER (target symbol → puzzle)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: DECODER TRAINING")
    print(f"{'='*60}")
    
    # Set up decoder-only training
    decoder_params = []
    for name, param in agent.named_parameters():
        if any(component in name for component in ['decoder', 'communication_embedding']):
            decoder_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=learning_rate)
    
    for epoch in range(decoder_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        
        indices = torch.randperm(len(puzzle_tensors))
        
        for idx in indices:
            i = idx.item()
            puzzle_tensor = puzzle_tensors[i]
            target_symbol_idx = targets[i]
            
            decoder_optimizer.zero_grad()
            
            # Create perfect one-hot symbol for the target
            perfect_symbol = torch.zeros(1, 1, num_comm_symbols, device=device)
            perfect_symbol[0, 0, target_symbol_idx] = 1.0
            
            # Forward pass through decoder with perfect target symbol
            reconstructed, reconstructed_logits, intermediates, confidences, size_logits = agent.decode_message_to_puzzle(
                perfect_symbol, temperature=0.1
            )
            
            # Reconstruction loss
            reconstruction_loss = compute_reconstruction_loss(
                reconstructed_logits, puzzle_tensor, agent.puzzle_symbols
            )
            
            # Size prediction loss
            target_size = (puzzle_tensor.size(1), puzzle_tensor.size(2))
            height_target = torch.tensor(target_size[0] - 1, device=device)
            width_target = torch.tensor(target_size[1] - 1, device=device)
            
            size_loss = (
                F.cross_entropy(size_logits[0], height_target.unsqueeze(0)) +
                F.cross_entropy(size_logits[1], width_target.unsqueeze(0))
            )
            
            # Regularization
            reg_loss = 0.0
            for param in decoder_params:
                reg_loss += param.pow(2.0).sum()
            
            total_loss_item = (
                reconstruction_loss +
                0.5 * size_loss +
                regularization_weight * reg_loss
            )
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            decoder_optimizer.step()
            
            # Track metrics
            reconstruction_acc = compute_reconstruction_accuracy(
                reconstructed_logits.argmax(dim=-1), puzzle_tensor
            )
            total_accuracy += reconstruction_acc
            total_loss += total_loss_item.item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(puzzle_tensors)
        avg_accuracy = total_accuracy / len(puzzle_tensors)
        
        history['encoder_loss'].append(0.0)  # Placeholder
        history['encoder_accuracy'].append(0.0)  # Placeholder
        history['decoder_loss'].append(avg_loss)
        history['decoder_accuracy'].append(avg_accuracy)
        history['unique_symbols'].append(0)  # Not relevant for decoder
        history['phase'].append('decoder')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Decoder Epoch {epoch+1}/{decoder_epochs}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}")
    
    print(f"\nDecoder training complete. Final accuracy: {avg_accuracy:.3f}")
    
    # =================================================================
    # FINAL EVALUATION: Test complete pipeline
    # =================================================================
    print(f"\n{'='*60}")
    print(f"FINAL PIPELINE EVALUATION")
    print(f"{'='*60}")
    
    # Re-enable all parameters for evaluation
    for param in agent.parameters():
        param.requires_grad = True
    
    encoder_correct = 0
    decoder_correct = 0
    pipeline_correct = 0
    
    print("\nTesting learned mappings:")
    for i, puzzle in enumerate(puzzle_tensors):
        if i >= 10:  # Limit output
            break
            
        with torch.no_grad():
            # Test encoder: puzzle → symbol
            symbols, symbol_logits, _ = agent.encode_puzzle_to_message(puzzle, temperature=0.1)
            pred_symbol = symbols[0, 0].argmax().item()
            target_symbol = targets[i]
            
            if pred_symbol == target_symbol:
                encoder_correct += 1
            
            # Test decoder: target symbol → puzzle
            perfect_symbol = torch.zeros(1, 1, num_comm_symbols, device=device)
            perfect_symbol[0, 0, target_symbol] = 1.0
            
            reconstructed, reconstructed_logits, _, _, _ = agent.decode_message_to_puzzle(
                perfect_symbol, temperature=0.1
            )
            
            decoder_acc = compute_reconstruction_accuracy(
                reconstructed_logits.argmax(dim=-1), puzzle
            )
            
            if decoder_acc > 0.8:  # High threshold for "correct"
                decoder_correct += 1
            
            # Test full pipeline: puzzle → symbol → puzzle
            full_reconstructed, full_logits, _, _, _ = agent.decode_message_to_puzzle(
                symbols, temperature=0.1
            )
            
            pipeline_acc = compute_reconstruction_accuracy(
                full_logits.argmax(dim=-1), puzzle
            )
            
            if pipeline_acc > 0.8:
                pipeline_correct += 1
            
            print(f"Puzzle {i}:")
            print(f"  Encoder: {pred_symbol} (target: {target_symbol}) {'✓' if pred_symbol == target_symbol else '✗'}")
            print(f"  Decoder: {decoder_acc:.3f} accuracy {'✓' if decoder_acc > 0.8 else '✗'}")
            print(f"  Pipeline: {pipeline_acc:.3f} accuracy {'✓' if pipeline_acc > 0.8 else '✗'}")
    
    test_count = min(10, len(puzzle_tensors))
    
    print(f"\nFinal Results:")
    print(f"  Encoder accuracy: {encoder_correct}/{test_count} = {encoder_correct/test_count:.3f}")
    print(f"  Decoder accuracy: {decoder_correct}/{test_count} = {decoder_correct/test_count:.3f}")
    print(f"  Pipeline accuracy: {pipeline_correct}/{test_count} = {pipeline_correct/test_count:.3f}")
    
    # Plot results
    plot_separate_pretraining_results(history, agent.agent_id, encoder_epochs)
    
    return history

def compute_reconstruction_loss(reconstructed_logits, target_puzzle, num_symbols):
    """Compute reconstruction loss between predicted and target puzzles."""
    # Get the overlapping region
    min_height = min(reconstructed_logits.size(1), target_puzzle.size(1))
    min_width = min(reconstructed_logits.size(2), target_puzzle.size(2))
    
    # Crop both to overlapping region
    pred_region = reconstructed_logits[:, :min_height, :min_width, :]
    target_region = target_puzzle[:, :min_height, :min_width]
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        pred_region.reshape(-1, num_symbols),
        target_region.reshape(-1),
        reduction='mean'
    )
    
    return loss

def plot_separate_pretraining_results(history, agent_name, encoder_epochs):
    """Plot results from separate encoder/decoder pre-training"""
    plt.figure(figsize=(20, 12))
    
    epochs = len(history['encoder_loss'])
    epoch_nums = list(range(1, epochs + 1))
    
    # Plot losses
    plt.subplot(2, 3, 1)
    encoder_losses = [loss if phase == 'encoder' else None for loss, phase in zip(history['encoder_loss'], history['phase'])]
    decoder_losses = [loss if phase == 'decoder' else None for loss, phase in zip(history['decoder_loss'], history['phase'])]
    
    # Filter out None values and create corresponding x-axis
    encoder_x = [i for i, loss in enumerate(encoder_losses, 1) if loss is not None]
    encoder_y = [loss for loss in encoder_losses if loss is not None]
    decoder_x = [i for i, loss in enumerate(decoder_losses, 1) if loss is not None]
    decoder_y = [loss for loss in decoder_losses if loss is not None]
    
    if encoder_y:
        plt.plot(encoder_x, encoder_y, label='Encoder Loss', linewidth=2, color='blue')
    if decoder_y:
        plt.plot(decoder_x, decoder_y, label='Decoder Loss', linewidth=2, color='red')
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title(f'{agent_name} Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 3, 2)
    encoder_accs = [acc if phase == 'encoder' else None for acc, phase in zip(history['encoder_accuracy'], history['phase'])]
    decoder_accs = [acc if phase == 'decoder' else None for acc, phase in zip(history['decoder_accuracy'], history['phase'])]
    
    encoder_acc_x = [i for i, acc in enumerate(encoder_accs, 1) if acc is not None]
    encoder_acc_y = [acc for acc in encoder_accs if acc is not None]
    decoder_acc_x = [i for i, acc in enumerate(decoder_accs, 1) if acc is not None]
    decoder_acc_y = [acc for acc in decoder_accs if acc is not None]
    
    if encoder_acc_y:
        plt.plot(encoder_acc_x, encoder_acc_y, label='Encoder Accuracy', linewidth=2, color='blue')
    if decoder_acc_y:
        plt.plot(decoder_acc_x, decoder_acc_y, label='Decoder Accuracy', linewidth=2, color='red')
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title(f'{agent_name} Training Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    
    # Plot unique symbols (encoder phase only)
    plt.subplot(2, 3, 3)
    unique_symbols = [symbols if phase == 'encoder' else None for symbols, phase in zip(history['unique_symbols'], history['phase'])]
    unique_x = [i for i, symbols in enumerate(unique_symbols, 1) if symbols is not None]
    unique_y = [symbols for symbols in unique_symbols if symbols is not None]
    
    if unique_y:
        plt.plot(unique_x, unique_y, label='Unique Symbols', linewidth=2, color='orange')
    
    plt.title(f'{agent_name} Unique Symbols Used (Encoder Phase)')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Combined loss plot
    plt.subplot(2, 3, 4)
    all_losses = []
    all_x = []
    colors = []
    for i, (enc_loss, dec_loss, phase) in enumerate(zip(history['encoder_loss'], history['decoder_loss'], history['phase'])):
        if phase == 'encoder' and enc_loss > 0:
            all_losses.append(enc_loss)
            all_x.append(i + 1)
            colors.append('blue')
        elif phase == 'decoder' and dec_loss > 0:
            all_losses.append(dec_loss)
            all_x.append(i + 1)
            colors.append('red')
    
    for i, (x, y, color) in enumerate(zip(all_x, all_losses, colors)):
        plt.scatter(x, y, c=color, alpha=0.6, s=20)
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title(f'{agent_name} Combined Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Encoder', 'Decoder', 'Phase Transition'])
    plt.grid(True)
    
    # Training phases visualization
    plt.subplot(2, 3, 5)
    phase_colors = ['blue' if p == 'encoder' else 'red' for p in history['phase']]
    plt.scatter(epoch_nums, [1] * len(epoch_nums), c=phase_colors, alpha=0.8, s=50)
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'{agent_name} Training Phases')
    plt.xlabel('Epoch')
    plt.ylabel('Phase')
    plt.yticks([1], ['Training'])
    plt.legend(['Encoder Phase', 'Decoder Phase'])
    plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Training Summary:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Encoder Epochs: {encoder_epochs}', fontsize=12)
    plt.text(0.1, 0.6, f'Decoder Epochs: {epochs - encoder_epochs}', fontsize=12)
    
    if encoder_acc_y:
        plt.text(0.1, 0.5, f'Final Encoder Acc: {encoder_acc_y[-1]:.3f}', fontsize=12)
    if decoder_acc_y:
        plt.text(0.1, 0.4, f'Final Decoder Acc: {decoder_acc_y[-1]:.3f}', fontsize=12)
    
    plt.text(0.1, 0.2, f'Separate training ensures\nconsistent mappings', fontsize=10, style='italic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{agent_name}_separate_pretraining.png', dpi=150, bbox_inches='tight')
    plt.close()


def compute_reconstruction_accuracy(pred_grid, target_puzzle):
    """Compute reconstruction accuracy handling size mismatches."""
    # Get the overlapping region
    min_height = min(pred_grid.size(1), target_puzzle.size(1))
    min_width = min(pred_grid.size(2), target_puzzle.size(2))
    
    # Crop both to overlapping region
    pred_region = pred_grid[:, :min_height, :min_width]
    target_region = target_puzzle[:, :min_height, :min_width]
    
    # Compute accuracy
    accuracy = (pred_region == target_region).float().mean().item()
    return accuracy

def plot_enhanced_pretraining_results(history, agent_name):
    """Plot enhanced results from end-to-end pre-training"""
    plt.figure(figsize=(20, 15))
    
    # Plot total loss
    plt.subplot(3, 2, 1)
    plt.plot(history['loss'], label='Total Loss', linewidth=2)
    plt.title(f'{agent_name} End-to-End Pre-training: Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot component losses
    plt.subplot(3, 2, 2)
    plt.plot(history['symbol_loss'], label='Symbol Loss', alpha=0.8)
    plt.plot(history['reconstruction_loss'], label='Reconstruction Loss', alpha=0.8)
    plt.title(f'{agent_name} Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot symbol accuracy
    plt.subplot(3, 2, 3)
    plt.plot(history['accuracy'], label='Symbol Accuracy', linewidth=2)
    plt.title(f'{agent_name} Symbol Assignment Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot reconstruction accuracy
    plt.subplot(3, 2, 4)
    plt.plot(history['reconstruction_accuracy'], label='Reconstruction Accuracy', linewidth=2, color='green')
    plt.title(f'{agent_name} Reconstruction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot unique symbols
    plt.subplot(3, 2, 5)
    plt.plot(history['unique_symbols'], label='Unique Symbols', linewidth=2, color='orange')
    plt.title(f'{agent_name} Unique Symbols Used')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    
    # Plot combined accuracy comparison
    plt.subplot(3, 2, 6)
    plt.plot(history['accuracy'], label='Symbol Accuracy', alpha=0.8)
    plt.plot(history['reconstruction_accuracy'], label='Reconstruction Accuracy', alpha=0.8)
    plt.title(f'{agent_name} Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{agent_name}_enhanced_pretraining.png', dpi=150)
    plt.close()

def plot_pretraining_results(history, agent_name):
    """Plot results from encoder pre-training"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(3, 1, 1)
    plt.plot(history['loss'])
    plt.title(f'{agent_name} Encoder Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(3, 1, 2)
    plt.plot(history['accuracy'])
    plt.title(f'{agent_name} Symbol Assignment Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Plot unique symbols
    plt.subplot(3, 1, 3)
    plt.plot(history['unique_symbols'])
    plt.title(f'{agent_name} Unique Symbols Used')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{agent_name}_pretraining.png')
    plt.close()

# Existing functions (unchanged)
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

def plot_training_metrics(metrics_history, accuracies_history, vocab_history=None, title="Training Metrics"):
    """
    PROGRESSIVE MODIFICATION: Added vocabulary progression plotting and improved size accuracy
    """
    plt.figure(figsize=(20, 15) if vocab_history else (15, 12))
    
    num_plots = 4 if vocab_history else 3
    
    # Plot loss
    plt.subplot(num_plots, 1, 1)
    losses = [m['total_loss'] for m in metrics_history if not np.isnan(m['total_loss'])]
    plt.plot(losses, label='Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot size accuracies (both distance-based and binary)
    plt.subplot(num_plots, 1, 2)
    plt.plot(accuracies_history['acc1_size'], label='Agent1 Size Acc (Distance)', alpha=0.8, linewidth=2)
    plt.plot(accuracies_history['acc1_size_binary'], label='Agent1 Size Acc (Binary)', alpha=0.6, linestyle='--')
    plt.plot(accuracies_history['acc2_size'], label='Agent2 Size Acc (Distance)', alpha=0.8, linewidth=2)
    plt.plot(accuracies_history['acc2_size_binary'], label='Agent2 Size Acc (Binary)', alpha=0.6, linestyle='--')
    plt.title(f'{title} - Size Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot content accuracies
    plt.subplot(num_plots, 1, 3)
    plt.plot(accuracies_history['acc1_content'], label='Agent1 Content Acc', alpha=0.7)
    plt.plot(accuracies_history['acc2_content'], label='Agent2 Content Acc', alpha=0.7)
    plt.title(f'{title} - Content Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
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
    plt.savefig('training_metrics.png')
    plt.close()

# Replace the pretrain_encoder function with this cross-agent version

def pretrain_cross_agent_communication(agent1, agent2, puzzles, device, 
                    epochs=None,           # For backward compatibility
                    encoder_epochs=None,   # NEW: Separate encoder epochs
                    decoder_epochs=None,   # NEW: Separate decoder epochs
                    learning_rate=0.001, 
                    diversity_weight=0.1,
                    regularization_weight=0.001,
                    visualization_frequency=50):
    """
    Simplified pre-training: Train Agent1's encoder and decoder separately, then copy to Agent2.
    
    Phase 1: Train Agent1_encoder (puzzle → target symbol)
    Phase 2: Train Agent1_decoder (target symbol → puzzle)  
    Phase 3: Copy Agent1's weights to Agent2
    
    Args:
        agent1, agent2: The agents to train
        puzzles: Training puzzles
        device: Training device
        epochs: Total epochs (for backward compatibility) - will be split equally if encoder_epochs/decoder_epochs not provided
        encoder_epochs: Number of epochs for encoder training (NEW)
        decoder_epochs: Number of epochs for decoder training (NEW)
        learning_rate: Learning rate
        diversity_weight: Weight for diversity loss
        regularization_weight: Weight for regularization
        visualization_frequency: How often to show training visualizations
    """
    import torch.nn.functional as F
    
    # Handle epoch parameters with backward compatibility
    if encoder_epochs is None and decoder_epochs is None:
        # Old behavior: split epochs equally
        if epochs is None:
            epochs = 300  # Default
        encoder_epochs = epochs // 2
        decoder_epochs = epochs - encoder_epochs
    elif encoder_epochs is None:
        # Only decoder_epochs specified
        if epochs is None:
            encoder_epochs = decoder_epochs // 2  # Default ratio
        else:
            encoder_epochs = epochs - decoder_epochs
    elif decoder_epochs is None:
        # Only encoder_epochs specified
        if epochs is None:
            decoder_epochs = encoder_epochs * 2  # Default: decoder gets more epochs
        else:
            decoder_epochs = epochs - encoder_epochs
    # If both encoder_epochs and decoder_epochs are specified, use them directly
    
    total_epochs = encoder_epochs + decoder_epochs
    
    print(f"\n===== Starting Customized Agent1 Pre-training =====")
    
    num_comm_symbols = agent1.current_comm_symbols
    print(f"Using {num_comm_symbols} communication symbols for pretraining")
    
    # Convert puzzles to tensors
    puzzle_tensors = [
        torch.tensor(puzzle.test_input, dtype=torch.long, device=device).unsqueeze(0)
        for puzzle in puzzles[:20]  # Limit to first 20 puzzles for efficiency
    ]
    
    # Assign target symbols to puzzles
    targets = {}
    for i, _ in enumerate(puzzle_tensors):
        targets[i] = i % min(num_comm_symbols, len(puzzle_tensors))
    
    print(f"Assigned {len(set(targets.values()))} unique target symbols to {len(puzzle_tensors)} puzzles")
    print("Target mappings:")
    for i in range(min(5, len(puzzle_tensors))):
        print(f"  Puzzle {i} → Symbol {targets[i]}")
    
    # Training history
    history = {
        'encoder_loss': [],
        'decoder_loss': [], 
        'encoder_accuracy': [],
        'decoder_accuracy': [],
        'pipeline_accuracy': [],
        'phase': []
    }
    
    print(f"\nCustomized Training Schedule:")
    print(f"Phase 1: Training Agent1 encoder for {encoder_epochs} epochs")
    print(f"Phase 2: Training Agent1 decoder for {decoder_epochs} epochs") 
    print(f"Total epochs: {total_epochs}")
    print(f"Phase 3: Copy Agent1 weights to Agent2")
    print(f"Reconstruction visualization every {visualization_frequency} epochs")
    
    # =================================================================
    # PHASE 1: TRAIN Agent1 ENCODER ONLY (puzzle → target symbol)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Agent1 ENCODER TRAINING ({encoder_epochs} epochs)")
    print(f"{'='*60}")
    
    # Set up encoder-only training for Agent1
    encoder_params = []
    for name, param in agent1.named_parameters():
        if any(component in name for component in ['encoder', 'embedding_system']):
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
        
        # Visualization flag
        show_visualization = (epoch % visualization_frequency == 0) or (epoch == encoder_epochs - 1)
        if show_visualization:
            print(f"\n--- Phase 1 Encoder Visualization (Epoch {epoch+1}) ---")
        
        for idx in indices:
            i = idx.item()
            puzzle_tensor = puzzle_tensors[i]
            
            encoder_optimizer.zero_grad()
            
            # Forward pass through Agent1's encoder only
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1
            )
            
            pred_symbol = symbols[0, 0].argmax().item()
            target = torch.tensor([targets[i]], device=device)
            
            # Symbol prediction loss only
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
            
            # Show encoder learning for first few puzzles during visualization epochs
            if show_visualization and i < 3:
                print(f"\nPuzzle {i} (Encoder Learning):")
                print(f"  Target symbol: {targets[i]}, Predicted: {pred_symbol} {'✓' if pred_symbol == targets[i] else '✗'}")
                print_grid(puzzle_tensor[0], f"  Input Puzzle")
                # Show symbol probabilities
                probs = F.softmax(symbol_logits[0, 0], dim=-1)
                top_probs, top_indices = torch.topk(probs, min(3, probs.size(0)))
                print(f"  Top symbol probabilities:")
                for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    symbol_num = agent1.puzzle_symbols + idx.item()
                    marker = " ← target" if idx.item() == targets[i] else ""
                    print(f"    {j+1}. Symbol {symbol_num}: {prob.item():.4f}{marker}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(puzzle_tensors)
        accuracy = correct / len(puzzle_tensors)
        
        history['encoder_loss'].append(avg_loss)
        history['encoder_accuracy'].append(accuracy)
        history['decoder_loss'].append(0.0)  # Placeholder
        history['decoder_accuracy'].append(0.0)  # Placeholder
        history['pipeline_accuracy'].append(0.0)  # Placeholder
        history['phase'].append('encoder')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Encoder Epoch {epoch+1}/{encoder_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
    
    print(f"\nPhase 1 complete. Final encoder accuracy: {accuracy:.3f}")
    
    # =================================================================
    # PHASE 2: TRAIN Agent1 DECODER ONLY (target symbol → puzzle)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Agent1 DECODER TRAINING ({decoder_epochs} epochs)")
    print(f"{'='*60}")
    
    # Set up decoder-only training for Agent1
    decoder_params = []
    for name, param in agent1.named_parameters():
        if any(component in name for component in ['decoder', 'communication_embedding']):
            decoder_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Keep Agent2 disabled
    for param in agent2.parameters():
        param.requires_grad = False
    
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=learning_rate)
    
    for epoch in range(decoder_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        
        indices = torch.randperm(len(puzzle_tensors))
        
        # Visualization flag
        show_visualization = (epoch % visualization_frequency == 0) or (epoch == decoder_epochs - 1)
        if show_visualization:
            print(f"\n--- Phase 2 Decoder Visualization (Epoch {epoch+1}) ---")
        
        for idx in indices:
            i = idx.item()
            puzzle_tensor = puzzle_tensors[i]
            target_symbol_idx = targets[i]
            
            decoder_optimizer.zero_grad()
            
            # Create perfect one-hot symbol for the target
            perfect_symbol = torch.zeros(1, 1, num_comm_symbols, device=device)
            perfect_symbol[0, 0, target_symbol_idx] = 1.0
            
            # Forward pass through Agent1's decoder with perfect target symbol
            reconstructed, reconstructed_logits, intermediates, confidences, size_logits = agent1.decode_message_to_puzzle(
                perfect_symbol, temperature=0.1
            )
            
            # Reconstruction loss
            reconstruction_loss = compute_reconstruction_loss(
                reconstructed_logits, puzzle_tensor, agent1.puzzle_symbols
            )
            
            # Size prediction loss
            target_size = (puzzle_tensor.size(1), puzzle_tensor.size(2))
            height_target = torch.tensor(target_size[0] - 1, device=device)
            width_target = torch.tensor(target_size[1] - 1, device=device)
            
            size_loss = (
                F.cross_entropy(size_logits[0], height_target.unsqueeze(0)) +
                F.cross_entropy(size_logits[1], width_target.unsqueeze(0))
            )
            
            # Regularization
            reg_loss = 0.0
            for param in decoder_params:
                reg_loss += param.pow(2.0).sum()
            
            total_loss_item = (
                reconstruction_loss +
                0.5 * size_loss +
                regularization_weight * reg_loss
            )
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            decoder_optimizer.step()
            
            # Track metrics
            reconstruction_acc = compute_reconstruction_accuracy(
                reconstructed_logits.argmax(dim=-1), puzzle_tensor
            )
            total_accuracy += reconstruction_acc
            total_loss += total_loss_item.item()
            
            # Show decoder learning for first few puzzles during visualization epochs
            if show_visualization and i < 3:
                print(f"\nPuzzle {i} (Decoder Learning):")
                print(f"  Input symbol: {target_symbol_idx}")
                print_grid(puzzle_tensor[0], f"  Target Puzzle")
                print_grid(reconstructed.argmax(dim=-1)[0], f"  Reconstructed")
                print(f"  Reconstruction accuracy: {reconstruction_acc:.3f}")
                print(f"  Decoder confidence scores: {[f'{c.item():.3f}' for c in confidences]}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(puzzle_tensors)
        avg_accuracy = total_accuracy / len(puzzle_tensors)
        
        history['encoder_loss'].append(0.0)  # Placeholder
        history['encoder_accuracy'].append(0.0)  # Placeholder
        history['decoder_loss'].append(avg_loss)
        history['decoder_accuracy'].append(avg_accuracy)
        history['pipeline_accuracy'].append(0.0)  # Placeholder
        history['phase'].append('decoder')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Decoder Epoch {epoch+1}/{decoder_epochs}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}")
    
    print(f"\nPhase 2 complete. Final decoder accuracy: {avg_accuracy:.3f}")

    # PHASE 3: COPY Agent1's weights to Agent2
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: COPYING Agent1 WEIGHTS TO Agent2")
    print(f"{'='*60}")
    
    with torch.no_grad():
        total_copied_params = 0
        
        # 1. Copy encoder weights
        print("Copying encoder weights...")
        encoder_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.encoder.named_parameters(), agent2.encoder.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                encoder_params_copied += param1.numel()
            else:
                print(f"  Warning: Shape mismatch for {name1}: {param1.shape} vs {param2.shape}")
        
        print(f"  ✓ Encoder weights copied ({encoder_params_copied:,} parameters)")
        total_copied_params += encoder_params_copied
        
        # 2. Copy decoder weights
        print("Copying decoder weights...")
        decoder_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.decoder.named_parameters(), agent2.decoder.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                decoder_params_copied += param1.numel()
            else:
                print(f"  Warning: Shape mismatch for {name1}: {param1.shape} vs {param2.shape}")
        
        print(f"  ✓ Decoder weights copied ({decoder_params_copied:,} parameters)")
        total_copied_params += decoder_params_copied
        
        # 3. Copy embedding system weights
        print("Copying embedding system weights...")
        embedding_params_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.embedding_system.named_parameters(), agent2.embedding_system.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                embedding_params_copied += param1.numel()
            else:
                print(f"  Warning: Shape mismatch for {name1}: {param1.shape} vs {param2.shape}")
        
        print(f"  ✓ Embedding system weights copied ({embedding_params_copied:,} parameters)")
        total_copied_params += embedding_params_copied
        
        # 4. Copy communication embeddings (current vocabulary only)
        print("Copying communication embeddings...")
        start_idx = agent1.puzzle_symbols
        end_idx = start_idx + agent1.current_comm_symbols
        
        agent2.communication_embedding.weight[start_idx:end_idx].copy_(
            agent1.communication_embedding.weight[start_idx:end_idx]
        )
        
        comm_params_copied = agent1.current_comm_symbols * agent1.communication_embedding.embedding_dim
        print(f"  ✓ Communication embeddings copied for symbols {start_idx}-{end_idx-1} ({comm_params_copied:,} parameters)")
        total_copied_params += comm_params_copied
        
        # 5. Verify vocabulary states match
        print("Verifying vocabulary states...")
        if agent1.current_comm_symbols != agent2.current_comm_symbols:
            print(f"  Warning: Vocabulary size mismatch! Agent1: {agent1.current_comm_symbols}, Agent2: {agent2.current_comm_symbols}")
            # Sync vocabulary states
            agent2.current_comm_symbols = agent1.current_comm_symbols
            agent2.current_seq_length = agent1.current_seq_length
            agent2.current_total_symbols = agent1.current_total_symbols
            agent2.communication_vocabulary = agent1.communication_vocabulary.copy()
            print(f"  ✓ Vocabulary states synchronized")
        else:
            print(f"  ✓ Vocabulary states match (vocab_size: {agent1.current_comm_symbols}, seq_length: {agent1.current_seq_length})")
        
        print(f"\nWeight copying complete!")
        print(f"Total parameters copied: {total_copied_params:,}")
        print(f"Both agents now have identical encoder/decoder mappings for current vocabulary")
        print(f"{'='*60}")
    
    # Quick verification test
    print("\nQuick verification test:")
    with torch.no_grad():
        test_puzzle = puzzle_tensors[0]
        
        # Test that both agents produce the same symbols for the same puzzle
        symbols1, _, _ = agent1.encode_puzzle_to_message(test_puzzle, temperature=0.1)
        symbols2, _, _ = agent2.encode_puzzle_to_message(test_puzzle, temperature=0.1)
        
        symbol1_idx = symbols1[0, 0].argmax().item()
        symbol2_idx = symbols2[0, 0].argmax().item()
        
        print(f"Same puzzle encoding test:")
        print(f"  Agent1 encodes to symbol: {symbol1_idx}")
        print(f"  Agent2 encodes to symbol: {symbol2_idx}")
        print(f"  Match: {'✓' if symbol1_idx == symbol2_idx else '✗'}")
        
        # Test that both agents decode the same symbol to similar puzzles
        target_symbol_idx = targets[0]
        perfect_symbol = torch.zeros(1, 1, num_comm_symbols, device=device)
        perfect_symbol[0, 0, target_symbol_idx] = 1.0
        
        recon1, _, _, _, _ = agent1.decode_message_to_puzzle(perfect_symbol, temperature=0.1)
        recon2, _, _, _, _ = agent2.decode_message_to_puzzle(perfect_symbol, temperature=0.1)
        
        decode_acc1 = compute_reconstruction_accuracy(recon1.argmax(dim=-1), test_puzzle)
        decode_acc2 = compute_reconstruction_accuracy(recon2.argmax(dim=-1), test_puzzle)
        
        print(f"Same symbol decoding test (symbol {target_symbol_idx}):")
        print(f"  Agent1 decode accuracy: {decode_acc1:.3f}")
        print(f"  Agent2 decode accuracy: {decode_acc2:.3f}")
        print(f"  Similar performance: {'✓' if abs(decode_acc1 - decode_acc2) < 0.1 else '✗'}")
    
    # =================================================================
    # FINAL EVALUATION: Test cross-agent communication
    # =================================================================
    print(f"\n{'='*60}")
    print(f"FINAL CROSS-AGENT EVALUATION")
    print(f"{'='*60}")
    
    # Re-enable all parameters for evaluation
    for param in agent1.parameters():
        param.requires_grad = True
    for param in agent2.parameters():
        param.requires_grad = True
    
    agent1_to_agent2_correct = 0
    agent2_to_agent1_correct = 0
    bidirectional_correct = 0
    
    print("\nTesting cross-agent communication:")
    for i, puzzle in enumerate(puzzle_tensors):
        if i >= 10:  # Limit output
            break
            
        with torch.no_grad():
            # Test Agent1 → Agent2 communication
            symbols1, _, _ = agent1.encode_puzzle_to_message(puzzle, temperature=0.1)
            reconstructed1, reconstructed_logits1, _, _, _ = agent2.decode_message_to_puzzle(
                symbols1, temperature=0.1
            )
            
            acc1to2 = compute_reconstruction_accuracy(
                reconstructed_logits1.argmax(dim=-1), puzzle
            )
            
            if acc1to2 > 0.8:
                agent1_to_agent2_correct += 1
            
            # Test Agent2 → Agent1 communication
            symbols2, _, _ = agent2.encode_puzzle_to_message(puzzle, temperature=0.1)
            reconstructed2, reconstructed_logits2, _, _, _ = agent1.decode_message_to_puzzle(
                symbols2, temperature=0.1
            )
            
            acc2to1 = compute_reconstruction_accuracy(
                reconstructed_logits2.argmax(dim=-1), puzzle
            )
            
            if acc2to1 > 0.8:
                agent2_to_agent1_correct += 1
            
            # Test bidirectional: Agent1 → Agent2 → Agent1
            symbols1to2, _, _ = agent1.encode_puzzle_to_message(puzzle, temperature=0.1)
            reconstructed1to2, _, _, _, _ = agent2.decode_message_to_puzzle(symbols1to2, temperature=0.1)
            symbols2to1, _, _ = agent2.encode_puzzle_to_message(reconstructed1to2.argmax(dim=-1), temperature=0.1)
            final_reconstructed, final_logits, _, _, _ = agent1.decode_message_to_puzzle(symbols2to1, temperature=0.1)
            
            bidirectional_acc = compute_reconstruction_accuracy(
                final_logits.argmax(dim=-1), puzzle
            )
            
            if bidirectional_acc > 0.8:
                bidirectional_correct += 1
            
            print(f"Puzzle {i}:")
            print(f"  Agent1→Agent2: {acc1to2:.3f} {'✓' if acc1to2 > 0.8 else '✗'}")
            print(f"  Agent2→Agent1: {acc2to1:.3f} {'✓' if acc2to1 > 0.8 else '✗'}")
            print(f"  Bidirectional: {bidirectional_acc:.3f} {'✓' if bidirectional_acc > 0.8 else '✗'}")
    
    test_count = min(10, len(puzzle_tensors))
    
    print(f"\nFinal Cross-Agent Results:")
    print(f"  Agent1→Agent2: {agent1_to_agent2_correct}/{test_count} = {agent1_to_agent2_correct/test_count:.3f}")
    print(f"  Agent2→Agent1: {agent2_to_agent1_correct}/{test_count} = {agent2_to_agent1_correct/test_count:.3f}")
    print(f"  Bidirectional: {bidirectional_correct}/{test_count} = {bidirectional_correct/test_count:.3f}")
    
    # Plot results - update the plotting function to show the custom epoch split
    plot_cross_agent_pretraining_results(history, encoder_epochs)
    
    return history


def plot_cross_agent_pretraining_results(history, encoder_epochs):
    """Plot results from cross-agent pre-training"""
    plt.figure(figsize=(20, 12))
    
    epochs = len(history['encoder_loss'])  # Changed from history['phase1_loss']
    epoch_nums = list(range(1, epochs + 1))
    
    # Plot losses
    plt.subplot(2, 3, 1)
    # Use the actual keys from the history dictionary
    encoder_losses = [loss if phase == 'encoder' else None for loss, phase in zip(history['encoder_loss'], history['phase'])]
    decoder_losses = [loss if phase == 'decoder' else None for loss, phase in zip(history['decoder_loss'], history['phase'])]
    
    # Filter out None values and create corresponding x-axis
    encoder_x = [i for i, loss in enumerate(encoder_losses, 1) if loss is not None]
    encoder_y = [loss for loss in encoder_losses if loss is not None]
    decoder_x = [i for i, loss in enumerate(decoder_losses, 1) if loss is not None]
    decoder_y = [loss for loss in decoder_losses if loss is not None]
    
    if encoder_y:
        plt.plot(encoder_x, encoder_y, label='Encoder Training', linewidth=2, color='blue')
    if decoder_y:
        plt.plot(decoder_x, decoder_y, label='Decoder Training', linewidth=2, color='red')
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Encoder/Decoder Pre-training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 3, 2)
    encoder_accs = [acc if phase == 'encoder' else None for acc, phase in zip(history['encoder_accuracy'], history['phase'])]
    decoder_accs = [acc if phase == 'decoder' else None for acc, phase in zip(history['decoder_accuracy'], history['phase'])]
    
    encoder_acc_x = [i for i, acc in enumerate(encoder_accs, 1) if acc is not None]
    encoder_acc_y = [acc for acc in encoder_accs if acc is not None]
    decoder_acc_x = [i for i, acc in enumerate(decoder_accs, 1) if acc is not None]
    decoder_acc_y = [acc for acc in decoder_accs if acc is not None]
    
    if encoder_acc_y:
        plt.plot(encoder_acc_x, encoder_acc_y, label='Encoder Accuracy', linewidth=2, color='blue')
    if decoder_acc_y:
        plt.plot(decoder_acc_x, decoder_acc_y, label='Decoder Accuracy', linewidth=2, color='red')
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Encoder/Decoder Pre-training Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    
    # Combined progress visualization
    plt.subplot(2, 3, 3)
    all_losses = []
    all_x = []
    colors = []
    for i, (enc_loss, dec_loss, phase) in enumerate(zip(history['encoder_loss'], history['decoder_loss'], history['phase'])):
        if phase == 'encoder' and enc_loss > 0:
            all_losses.append(enc_loss)
            all_x.append(i + 1)
            colors.append('blue')
        elif phase == 'decoder' and dec_loss > 0:
            all_losses.append(dec_loss)
            all_x.append(i + 1)
            colors.append('red')
    
    for i, (x, y, color) in enumerate(zip(all_x, all_losses, colors)):
        plt.scatter(x, y, c=color, alpha=0.6, s=20)
    
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    plt.title('Combined Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Encoder', 'Decoder', 'Phase Transition'])
    plt.grid(True)
    
    # Training phases visualization
    plt.subplot(2, 3, 4)
    phase_colors = ['blue' if p == 'encoder' else 'red' for p in history['phase']]
    plt.scatter(epoch_nums, [1] * len(epoch_nums), c=phase_colors, alpha=0.8, s=50)
    plt.axvline(x=encoder_epochs, color='gray', linestyle='--', alpha=0.7)
    plt.title('Training Phases')
    plt.xlabel('Epoch')
    plt.ylabel('Phase')
    plt.yticks([1], ['Training'])
    plt.legend(['Encoder Phase', 'Decoder Phase'])
    plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.8, f'Training Summary:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Encoder Epochs: {encoder_epochs}', fontsize=12)
    plt.text(0.1, 0.6, f'Decoder Epochs: {epochs - encoder_epochs}', fontsize=12)
    
    if encoder_acc_y:
        plt.text(0.1, 0.5, f'Final Encoder Acc: {encoder_acc_y[-1]:.3f}', fontsize=12)
    if decoder_acc_y:
        plt.text(0.1, 0.4, f'Final Decoder Acc: {decoder_acc_y[-1]:.3f}', fontsize=12)
    
    plt.text(0.1, 0.2, f'Separate training ensures\nconsistent mappings', fontsize=10, style='italic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Architecture diagram
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.9, 'Encoder/Decoder Architecture', fontsize=14, fontweight='bold', ha='center')
    
    # Phase 1 diagram
    plt.text(0.5, 0.8, 'Phase 1:', fontsize=12, fontweight='bold', ha='center')
    plt.text(0.5, 0.75, 'Puzzle → Encoder → Target Symbol', fontsize=10, ha='center')
    
    # Phase 2 diagram  
    plt.text(0.5, 0.6, 'Phase 2:', fontsize=12, fontweight='bold', ha='center')
    plt.text(0.5, 0.55, 'Target Symbol → Decoder → Puzzle', fontsize=10, ha='center')
    
    # Benefits
    plt.text(0.5, 0.4, 'Benefits:', fontsize=12, fontweight='bold', ha='center')
    plt.text(0.5, 0.35, '• Consistent symbol mappings', fontsize=10, ha='center')
    plt.text(0.5, 0.3, '• Separate encoder/decoder training', fontsize=10, ha='center')
    plt.text(0.5, 0.25, '• Better initialization for joint training', fontsize=10, ha='center')
    plt.text(0.5, 0.2, '• Reduced training instability', fontsize=10, ha='center')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cross_agent_pretraining.png', dpi=150, bbox_inches='tight')
    plt.close()

# The remaining functions (communication debugging, training cycle, etc.) remain unchanged

# The main function now includes the pre-training phase
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # PROGRESSIVE MODIFICATION: Create progressive agents with maximum capacity
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=12,      # PROGRESSIVE: Maximum symbols
        puzzle_symbols=10,
        max_seq_length=1,    # PROGRESSIVE: Maximum sequence length
        sender_scale=1.0
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=12,      # PROGRESSIVE: Maximum symbols
        puzzle_symbols=10,
        max_seq_length=1,    # PROGRESSIVE: Maximum sequence length
        sender_scale=1.0 
    ).to(device)
    
    # PROGRESSIVE MODIFICATION: Create progressive trainer with expansion settings
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=1e-7,
        device=device,
        expansion_frequency=10000,  # PROGRESSIVE: Expand every 200 cycles
        symbols_per_expansion=2,   # PROGRESSIVE: Add 2 symbols per expansion
        length_per_expansion=1     # PROGRESSIVE: Add 1 to sequence length per expansion
    )
    
    # Load ARC puzzles (unchanged)
    arc_file_path = 'arc-agi_test_challenges.json'
    test_puzzles = load_arc_puzzles(arc_file_path)
    print(f"\nLoaded {len(test_puzzles)} total examples from ARC dataset")
    # selected_puzzles = [test_puzzles[0], test_puzzles[6]]
    selected_puzzles = [test_puzzles[0], test_puzzles[47]]
    # selected_puzzles = [test_puzzles[i] for i in range(0, len(test_puzzles), 10)][:10]
    print(f"Selected {len(selected_puzzles)} diverse puzzles for training")
    
    # PROGRESSIVE MODIFICATION: Show initial vocabulary state
    print("\n" + "="*60)
    print("INITIAL VOCABULARY STATE")
    print("="*60)
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)

    encoder_pretrain_epochs = 0   # Encoder training epochs
    decoder_pretrain_epochs = 0  # Decoder training epochs (more as you need)

    if encoder_pretrain_epochs > 0 or decoder_pretrain_epochs > 0:
        cross_agent_history = pretrain_cross_agent_communication(
            sender, receiver, selected_puzzles, device, 
            encoder_epochs=encoder_pretrain_epochs,
            decoder_epochs=decoder_pretrain_epochs,
            learning_rate=0.0001,
            diversity_weight=0.2,
            regularization_weight=0.001
        )
        
        print(f"Pre-training completed:")
        print(f"  Encoder trained for {encoder_pretrain_epochs} epochs")
        print(f"  Decoder trained for {decoder_pretrain_epochs} epochs")
        print(f"  Total pre-training epochs: {encoder_pretrain_epochs + decoder_pretrain_epochs}")
    
    # # PHASE 1: Supervised pre-training (optional, can be disabled)
    # pretrain_epochs = 1000  # Set to 0 to skip pre-training
    
    # if pretrain_epochs > 0:
    #     sender_history = pretrain_encoder(
    #         sender, selected_puzzles, device, 
    #         epochs=pretrain_epochs,
    #         learning_rate=0.001,
    #         diversity_weight=0.2,
    #         regularization_weight=0.001
    #     )
        
    #     receiver_history = pretrain_encoder(
    #         receiver, selected_puzzles, device, 
    #         epochs=pretrain_epochs,
    #         learning_rate=0.001,
    #         diversity_weight=0.2,
    #         regularization_weight=0.001
    #     )
    
    # PHASE 2: Main progressive communication training
    total_cycles = 10000  # This will see multiple expansions
    
    print(f"\n--- Starting Progressive Joint Training ({total_cycles} cycles) ---")
    trainer.set_training_mode("joint")
    
    # Initialize histories and trackers
    metrics_history = []
    acc1_size_history = []
    acc1_content_history = []
    acc2_size_history = []
    acc2_content_history = []
    acc1_size_binary_history = [] 
    acc2_size_binary_history = [] 
    
    # PROGRESSIVE MODIFICATION: Track vocabulary progression
    vocab_history = {
        'cycles': [],
        'vocab_sizes': [],
        'seq_lengths': [],
        'expansion_cycles': []
    }
    
    # Initialize moving averages
    ma_window = 50
    acc1_size_ma = MovingAverage(ma_window)
    acc1_content_ma = MovingAverage(ma_window)
    acc2_size_ma = MovingAverage(ma_window)
    acc2_content_ma = MovingAverage(ma_window)
    
    puzzle_to_symbols = {}
    
    # PROGRESSIVE MODIFICATION: Enhanced logging
    with open('progressive_training_log.txt', 'w') as log_file:
        log_file.write("Progressive Communication Training Log\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Expansion every {trainer.expansion_frequency} cycles\n")
        log_file.write(f"Add {trainer.symbols_per_expansion} symbols and {trainer.length_per_expansion} sequence length per expansion\n")
        log_file.write("Size Accuracy: Now using distance-based metric (1.0 = perfect, decreases with distance)\n")
        log_file.write("="*50 + "\n\n")
        
        for cycle in range(total_cycles):
            print(f"\nCycle {cycle + 1}/{total_cycles}")
            
            # PROGRESSIVE MODIFICATION: Track vocabulary progression
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
            
            cycle_metrics = train_cycle(cycle, trainer, selected_puzzles, device, puzzle_to_symbols)
            
            # Update histories and moving averages (UPDATED)
            accuracy_histories = update_metrics(cycle_metrics, metrics_history, 
                        acc1_size_ma, acc1_content_ma, acc2_size_ma, acc2_content_ma,
                        acc1_size_history, acc1_content_history, acc2_size_history, acc2_content_history)
            
            # PROGRESSIVE MODIFICATION: Enhanced logging with vocabulary info
            log_cycle_metrics(log_file, cycle, cycle_metrics, acc1_size_ma, acc1_content_ma, 
                        acc2_size_ma, acc2_content_ma, sender)
            
            # Plot metrics periodically
            if (cycle + 1) % 10 == 0:
                plot_training_metrics(metrics_history, accuracy_histories, vocab_history, 
                                    title=f"Progressive Training Metrics (Cycle {cycle+1})")

    # Add a final detailed metrics summary
    print("\n" + "="*60)
    print("FINAL TRAINING SUMMARY")
    print("="*60)

    # Show recent size errors
    recent_metrics = metrics_history[-50:]  # Last 50 steps
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

    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)
    
    print("\nProgressive training complete! Check progressive_training_log.txt for details")

# Include remaining functions from train.py
def train_cycle(cycle, trainer, test_puzzles, device, puzzle_to_symbols):
    """Train for one cycle through all puzzles."""
    cycle_metrics = []
    
    visualization_frequency = 10
    
    for puzzle_idx, puzzle in enumerate(test_puzzles):
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
        
        # PROGRESSIVE MODIFICATION: Show communication debug with vocabulary info
        if cycle % visualization_frequency == 0:
            print(f"\n--- Visualization for Cycle {cycle}, Puzzle {puzzle_idx} ---")
            print_communication_debug(puzzle_tensor, trainer.agent1, trainer.agent2)
        
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
    
    return cycle_metrics

def update_metrics(cycle_metrics, metrics_history, 
                   acc1_size_ma, acc1_content_ma, acc2_size_ma, acc2_content_ma,
                   acc1_size_history, acc1_content_history, acc2_size_history, acc2_content_history):
    """Update all metrics tracking arrays and moving averages"""
    metrics_history.extend(cycle_metrics)
    
    # Also create separate tracking for binary accuracy
    acc1_size_binary_history = []
    acc2_size_binary_history = []
    
    # Update moving averages
    for metrics in cycle_metrics:
        acc1_size_ma.update(metrics['agent1_size_accuracy'])  # Distance-based
        acc1_content_ma.update(metrics['agent1_content_accuracy'])
        acc2_size_ma.update(metrics['agent2_size_accuracy'])  # Distance-based
        acc2_content_ma.update(metrics['agent2_content_accuracy'])
        
        # Store history
        acc1_size_history.append(acc1_size_ma.get_average())
        acc1_content_history.append(acc1_content_ma.get_average())
        acc2_size_history.append(acc2_size_ma.get_average())
        acc2_content_history.append(acc2_content_ma.get_average())
        
        # Also track binary accuracy for comparison
        acc1_size_binary_history.append(metrics['agent1_size_binary'])
        acc2_size_binary_history.append(metrics['agent2_size_binary'])
    
    return {
        'acc1_size': acc1_size_history,
        'acc1_content': acc1_content_history,
        'acc2_size': acc2_size_history,
        'acc2_content': acc2_content_history,
        'acc1_size_binary': acc1_size_binary_history,
        'acc2_size_binary': acc2_size_binary_history
    }


def log_cycle_metrics(log_file, cycle, cycle_metrics, acc1_size_ma, acc1_content_ma, 
                     acc2_size_ma, acc2_content_ma, agent):
    """PROGRESSIVE MODIFICATION: Enhanced logging with vocabulary info and detailed size metrics"""
    avg_metrics = {
        'total_loss': np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])]),
        'avg_size_error1': np.mean([m.get('agent1_size_error', 0) for m in cycle_metrics]),
        'avg_size_error2': np.mean([m.get('agent2_size_error', 0) for m in cycle_metrics]),
        'training_mode': cycle_metrics[0]['training_mode'] if 'training_mode' in cycle_metrics[0] else 'unknown'
    }
    
    log_file.write(
        f"Cycle {cycle + 1}: " + 
        f"Loss={avg_metrics['total_loss']:.4f}, " +
        f"Mode={avg_metrics['training_mode']}, " +
        f"Vocab={agent.current_comm_symbols}, " +
        f"SeqLen={agent.current_seq_length}, " +
        f"Size_Acc1={acc1_size_ma.get_average():.3f}, " +
        f"Content_Acc1={acc1_content_ma.get_average():.3f}, " +
        f"Size_Acc2={acc2_size_ma.get_average():.3f}, " +
        f"Content_Acc2={acc2_content_ma.get_average():.3f}, " +
        f"SizeErr1={avg_metrics['avg_size_error1']:.1f}, " +
        f"SizeErr2={avg_metrics['avg_size_error2']:.1f}\n"
    )
    log_file.flush()

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

def print_communication_debug(puzzle_tensor, sender, receiver):
    """
    PROGRESSIVE MODIFICATION: Enhanced debugging for progressive agents with symbol probabilities
    """
    import torch.nn.functional as F
    
    print("\n  Communication Example:")
    print_grid(puzzle_tensor[0], "Original Puzzle")
    
    # PROGRESSIVE MODIFICATION: Show vocabulary status
    print(f"\nCurrent Vocabulary Status:")
    print(f"  Sender: {sender.current_comm_symbols} symbols, {sender.current_seq_length} length")
    print(f"  Receiver: {receiver.current_comm_symbols} symbols, {receiver.current_seq_length} length")
    
    # Sender to Receiver communication
    print("\nSender → Receiver:")
    symbols1, symbol_logits1, stats1 = sender.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols1, "Sender")
    
    # NEW: Show encoder symbol probabilities
    print("\nSender Encoder Symbol Probabilities:")
    for pos in range(stats1['current_seq_length']):
        # Get the symbol logits for this position
        pos_logits = symbol_logits1[0, pos]  # [num_comm_symbols]
        pos_probs = F.softmax(pos_logits, dim=-1)
        
        # Get the selected symbol index
        selected_symbol_idx = symbols1[0, pos].argmax().item()
        selected_prob = pos_probs[selected_symbol_idx].item()
        
        # Convert to actual symbol number (add puzzle_symbols offset)
        actual_symbol = sender.puzzle_symbols + selected_symbol_idx
        
        print(f"  Position {pos}: Symbol {actual_symbol} with probability {selected_prob:.4f}")
        
        # Show top 3 probabilities for this position if there are multiple options
        top_probs, top_indices = torch.topk(pos_probs, min(3, pos_probs.size(0)))
        if len(top_probs) > 1:
            print(f"    Top probabilities:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                symbol_num = sender.puzzle_symbols + idx.item()
                marker = " ← selected" if idx.item() == selected_symbol_idx else ""
                print(f"      {i+1}. Symbol {symbol_num}: {prob.item():.4f}{marker}")
    
    # PROGRESSIVE MODIFICATION: Show position-based usage
    print(f"\nPosition-based symbol usage:")
    for pos in range(stats1['current_seq_length']):
        symbol_idx = symbols1[0, pos].argmax().item()
        actual_symbol = sender.puzzle_symbols + symbol_idx
        print(f"  Position {pos}: Uses symbol {actual_symbol}")
    
    reconstructed1, grid_logits1, intermediates1, confidences1, size_logits1 = receiver.decode_message_to_puzzle(
        symbols1, temperature=0.1
    )
    
    print("\nReceiver Decoder Confidence Scores:")
    for i, conf in enumerate(confidences1):
        print(f"  Step {i+1}: {conf.item():.4f}")
    
    print_grid(reconstructed1.argmax(dim=-1)[0], "Sender→Receiver Reconstruction")

    # Receiver to Sender communication
    print("\nReceiver → Sender:")
    symbols2, symbol_logits2, stats2 = receiver.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols2, "Receiver")
    
    # NEW: Show encoder symbol probabilities for receiver
    print("\nReceiver Encoder Symbol Probabilities:")
    for pos in range(stats2['current_seq_length']):
        # Get the symbol logits for this position
        pos_logits = symbol_logits2[0, pos]  # [num_comm_symbols]
        pos_probs = F.softmax(pos_logits, dim=-1)
        
        # Get the selected symbol index
        selected_symbol_idx = symbols2[0, pos].argmax().item()
        selected_prob = pos_probs[selected_symbol_idx].item()
        
        # Convert to actual symbol number (add puzzle_symbols offset)
        actual_symbol = receiver.puzzle_symbols + selected_symbol_idx
        
        print(f"  Position {pos}: Symbol {actual_symbol} with probability {selected_prob:.4f}")
        
        # Show top 3 probabilities for this position if there are multiple options
        top_probs, top_indices = torch.topk(pos_probs, min(3, pos_probs.size(0)))
        if len(top_probs) > 1:
            print(f"    Top probabilities:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                symbol_num = receiver.puzzle_symbols + idx.item()
                marker = " ← selected" if idx.item() == selected_symbol_idx else ""
                print(f"      {i+1}. Symbol {symbol_num}: {prob.item():.4f}{marker}")
    
    reconstructed2, grid_logits2, intermediates2, confidences2, size_logits2 = sender.decode_message_to_puzzle(
        symbols2, temperature=0.1
    )
    
    print("\nSender Decoder Confidence Scores:")
    for i, conf in enumerate(confidences2):
        print(f"  Step {i+1}: {conf.item():.4f}")
    
    print_grid(reconstructed2.argmax(dim=-1)[0], "Receiver→Sender Reconstruction")
if __name__ == "__main__":
    main()