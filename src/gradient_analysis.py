import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import Agent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

class GradientPathTracker:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        self.gradient_magnitudes = defaultdict(list)
        self.hooks = []
        
    def _register_hook(self, module, name):
        if hasattr(module, 'weight') and module.weight is not None:
            def hook(grad):
                self.gradient_magnitudes[name].append(grad.abs().mean().item())
                return grad
            
            handle = module.weight.register_hook(hook)
            self.hooks.append(handle)
    
    def attach_hooks(self):
        # Sender pathway
        for name, module in self.sender.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                self._register_hook(module, f"sender.{name}")
        self._register_hook(self.sender.communication_embedding, "sender.comm_embedding")
        
        # Receiver pathway
        for name, module in self.receiver.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                self._register_hook(module, f"receiver.{name}")
        self._register_hook(self.receiver.communication_embedding, "receiver.comm_embedding")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create two agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,
        puzzle_symbols=10,
        max_seq_length=5
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=20,
        puzzle_symbols=10,
        max_seq_length=5
    ).to(device)
    
    # Create gradient tracker
    tracker = GradientPathTracker(sender, receiver)
    tracker.attach_hooks()
    
    # Create sample data
    puzzle = torch.randint(0, 10, (1, 5, 5), device=device)
    
    print("\nRunning forward and backward passes...")
    
    # Forward pass through sender
    message, symbol_logits, stats = sender.encode_puzzle_to_message(puzzle)
    print("\nMessage shape:", message.shape)
    print("Message sparsity:", (message.max(dim=-1)[0] > 0.5).float().mean().item())
    
    # Forward pass through receiver with intermediate outputs
    reconstructed, logits, intermediate_logits, confidence_scores, _ = receiver.decode_message_to_puzzle(
        message,
        target_size=(5, 5)
    )
    
    print(f"\nNumber of intermediate outputs: {len(intermediate_logits)}")
    print("\nConfidence progression:")
    for i, conf in enumerate(confidence_scores):
        print(f"Step {i} confidence: {conf.item():.4f}")
    
    # Compute reconstruction losses
    step_weights = torch.linspace(0.5, 1.0, len(intermediate_logits), device=device)
    recon_losses = []
    
    for step_output in intermediate_logits:
        step_loss = F.cross_entropy(
            step_output.reshape(-1, receiver.puzzle_symbols),
            puzzle.reshape(-1)
        )
        recon_losses.append(step_loss)
    
    # Final reconstruction loss
    final_loss = F.cross_entropy(
        logits.reshape(-1, receiver.puzzle_symbols),
        puzzle.reshape(-1)
    )
    
    # Stack confidences
    confidences = torch.cat(confidence_scores, dim=1)
    
    # Compute direct feedback losses
    confidence_growth_loss = F.relu(confidences[:, :-1] - confidences[:, 1:]).mean()
    
    # Compute actual accuracy
    with torch.no_grad():
        actual_accuracy = (reconstructed.argmax(dim=-1) == puzzle).float().mean()
    confidence_accuracy_loss = F.mse_loss(confidences[:, -1], actual_accuracy.expand_as(confidences[:, -1]))
    
    # Message entropy loss
    message_entropy = -(message * torch.log(message + 1e-10)).sum(dim=-1).mean()
    entropy_confidence_loss = F.mse_loss(message_entropy, 1 - confidences[:, -1].mean())
    
    # Combine all losses
    reconstruction_loss = sum(w * l for w, l in zip(step_weights, recon_losses)) + final_loss
    direct_feedback_loss = (
        0.1 * confidence_growth_loss +
        0.1 * confidence_accuracy_loss +
        0.1 * entropy_confidence_loss
    )
    
    total_loss = reconstruction_loss + direct_feedback_loss
    
    print(f"\nLoss components:")
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"Confidence growth loss: {confidence_growth_loss.item():.4f}")
    print(f"Confidence accuracy loss: {confidence_accuracy_loss.item():.4f}")
    print(f"Entropy confidence loss: {entropy_confidence_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Analyze gradients
    print("\nGradient magnitudes along the backward path:")
    print("-" * 50)
    
    # Group components
    receiver_grads = {k: v for k, v in tracker.gradient_magnitudes.items() if k.startswith('receiver')}
    sender_grads = {k: v for k, v in tracker.gradient_magnitudes.items() if k.startswith('sender')}
    
    # Sort by gradient magnitude
    def print_sorted_grads(grads, title):
        if grads:
            print(f"\n{title}:")
            sorted_grads = sorted(grads.items(), key=lambda x: np.mean(x[1]), reverse=True)
            for name, magnitudes in sorted_grads:
                print(f"{name:40s}: {np.mean(magnitudes):.12f}")
    
    print_sorted_grads(receiver_grads, "Receiver Gradients (from output to embeddings)")
    print_sorted_grads(sender_grads, "Sender Gradients (from embeddings to input)")
    
    # Visualize gradient flow
    plt.figure(figsize=(15, 8))

    # Define a custom order that reflects the network's computational graph
    layer_order = [
        # Receiver decoder path (output to input)
        "receiver.decoder.output",  # Final linear projection
        "receiver.decoder.refinement_layers",  # Refinement steps
        "receiver.decoder.pos_encoder",  # Positional encoding
        "receiver.decoder.input_projection",  # Input projection
        "receiver.communication_embedding",  # Communication embedding
        
        # Receiver encoder path (output to input)
        "receiver.encoder.length_head",  # Length prediction
        "receiver.encoder.embedding_predictor",  # Embedding prediction
        "receiver.encoder.intermediate",  # Intermediate processing
        "receiver.encoder.encoder_norm",  # Encoder normalization
        "receiver.encoder.encoder",  # Transformer encoder layers
        "receiver.encoder.pos_encoder",  # Positional encoding
        
        # Similar path for sender
        "sender.decoder.output",
        "sender.decoder.refinement_layers",
        "sender.decoder.pos_encoder",
        "sender.decoder.input_projection",
        "sender.communication_embedding",
        
        "sender.encoder.length_head",
        "sender.encoder.embedding_predictor", 
        "sender.encoder.intermediate",
        "sender.encoder.encoder_norm",
        "sender.encoder.encoder",
        "sender.encoder.pos_encoder"
    ]

    # Debugging: Print all available gradient magnitudes
    print("\nAll available gradient magnitudes:")
    for name, magnitudes in tracker.gradient_magnitudes.items():
        print(f"{name}: {np.mean(magnitudes)}")

    # Collect gradients in the specified order
    ordered_grads = []
    for layer_name in layer_order:
        matching_layers = [
            (name, np.mean(mags)) 
            for name, mags in tracker.gradient_magnitudes.items() 
            if layer_name in name
        ]
        
        if matching_layers:
            # If multiple layers match, use the first one
            ordered_grads.extend(matching_layers)

    # Add any remaining layers not in the predefined order
    remaining_layers = set(tracker.gradient_magnitudes.keys()) - set(name for name, _ in ordered_grads)

    for layer in remaining_layers:
        ordered_grads.append((layer, np.mean(tracker.gradient_magnitudes[layer])))

    # Plot
    plt.figure(figsize=(15, 8))
    names = [g[0] for g in ordered_grads]
    values = [g[1] for g in ordered_grads]

    plt.barh(range(len(values)), values)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xscale('log')
    plt.xlabel('Gradient Magnitude (log scale)')
    plt.title('Gradient Magnitudes Through Network (Computational Graph Order)')
    plt.tight_layout()
    plt.savefig('gradient_path.png')
    
    # Print specific embedding gradients
    print("\nSpecific embedding gradient comparison:")
    print(f"Sender comm embedding gradient:   {np.mean(tracker.gradient_magnitudes['sender.comm_embedding']):.12f}")
    print(f"Receiver comm embedding gradient: {np.mean(tracker.gradient_magnitudes['receiver.comm_embedding']):.12f}")
    
    # Clean up
    tracker.remove_hooks()
    plt.close()
    
    print("\nGradient analysis complete! Check 'gradient_path.png' for visualization.")

if __name__ == "__main__":
    main()