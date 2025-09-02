import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from agent import ProgressiveAgent
from puzzle import Puzzle
import numpy as np

class ProgressiveCommunicationTrainer:
    def __init__(
        self,
        agent1: ProgressiveAgent,
        agent2: ProgressiveAgent,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        expansion_frequency: int = 200,  # Expand every N cycles
        symbols_per_expansion: int = 2,   # Add N symbols per expansion
        length_per_expansion: int = 1,    # Add N sequence length per expansion
        sync_frequency: int = 50          # NEW: Synchronize every N cycles
    ):
        self.agent1 = agent1.to(device)
        self.agent2 = agent2.to(device)
        self.device = device
        
        # Progressive training parameters
        self.expansion_frequency = expansion_frequency
        self.symbols_per_expansion = symbols_per_expansion
        self.length_per_expansion = length_per_expansion
        self.cycle_count = 0
        
        # NEW: Synchronization parameters
        self.sync_frequency = sync_frequency
        self.last_sync_cycle = 0

        # Copy Agent 1's puzzle symbol embeddings to Agent 2
        with torch.no_grad():
            puzzle_embeddings = agent1.embedding_system.symbol_embedding.weight[:agent1.puzzle_symbols].clone()
            agent2.embedding_system.symbol_embedding.weight[:agent2.puzzle_symbols] = puzzle_embeddings

        # Initialize optimizers
        self.opt1 = optim.Adam([
            {'params': agent1.embedding_system.parameters(), 'lr': learning_rate},
            {'params': agent1.encoder.parameters(), 'lr': learning_rate},
            {'params': agent1.decoder.parameters(), 'lr': learning_rate},
            {'params': agent1.communication_embedding.parameters(), 'lr': learning_rate}
        ])

        self.opt2 = optim.Adam([
            {'params': agent2.embedding_system.parameters(), 'lr': learning_rate},
            {'params': agent2.encoder.parameters(), 'lr': learning_rate},
            {'params': agent2.decoder.parameters(), 'lr': learning_rate},
            {'params': agent2.communication_embedding.parameters(), 'lr': learning_rate}
        ])

        self.in_initial_phase = True
        
        # Loss criteria
        self.symbol_criterion = nn.CrossEntropyLoss()
        self.grid_criterion = nn.CrossEntropyLoss()
        
        # Training mode tracking
        self.training_mode = "joint"
    
    def should_synchronize(self) -> bool:
        """Check if it's time to synchronize agent parameters"""
        return (self.cycle_count > 0 and 
                self.cycle_count - self.last_sync_cycle >= self.sync_frequency)
    
    def _sync_module_recursive(self, module1, module2, module_name=""):
        """Recursively synchronize all parameters in a module"""
        synced_count = 0
        
        # Sync direct parameters
        for name, param1 in module1.named_parameters(recurse=False):
            if hasattr(module2, name.split('.')[-1]):
                param2 = getattr(module2, name.split('.')[-1])
                if hasattr(param2, 'copy_'):
                    param2.copy_(param1)
                    synced_count += param1.numel()
        
        # Sync buffers (like running means in batch norm)
        for name, buffer1 in module1.named_buffers(recurse=False):
            if hasattr(module2, name.split('.')[-1]):
                buffer2 = getattr(module2, name.split('.')[-1])
                if hasattr(buffer2, 'copy_'):
                    buffer2.copy_(buffer1)
                    synced_count += buffer1.numel()
        
        # Recursively sync child modules
        for name, child1 in module1.named_children():
            if hasattr(module2, name):
                child2 = getattr(module2, name)
                synced_count += self._sync_module_recursive(child1, child2, f"{module_name}.{name}")
        
        return synced_count

    def synchronize_agents(self):
        """
        Synchronize ALL agent parameters from agent1 to agent2.
        This ensures both agents have identical representations and processing.
        """
        print(f"\n{'='*60}")
        print(f"SYNCHRONIZING AGENTS AT CYCLE {self.cycle_count}")
        print(f"{'='*60}")
        
        with torch.no_grad():
            total_synced_params = 0
            
            # 1. Synchronize puzzle embedding system
            print("Synchronizing puzzle embedding system...")
            embedding_params = self._sync_module_recursive(
                self.agent1.embedding_system, 
                self.agent2.embedding_system, 
                "embedding_system"
            )
            total_synced_params += embedding_params
            print(f"  ✓ Puzzle embedding system synchronized ({embedding_params:,} parameters)")
            
            # 2. Synchronize encoder
            print("Synchronizing encoder...")
            encoder_params = self._sync_module_recursive(
                self.agent1.encoder, 
                self.agent2.encoder, 
                "encoder"
            )
            total_synced_params += encoder_params
            print(f"  ✓ Encoder synchronized ({encoder_params:,} parameters)")
            
            # 3. Synchronize decoder  
            print("Synchronizing decoder...")
            decoder_params = self._sync_module_recursive(
                self.agent1.decoder, 
                self.agent2.decoder, 
                "decoder"
            )
            total_synced_params += decoder_params
            print(f"  ✓ Decoder synchronized ({decoder_params:,} parameters)")
            
            # 4. Synchronize communication embeddings (current vocabulary only)
            print("Synchronizing communication embeddings...")
            current_comm_symbols = self.agent1.current_comm_symbols
            if current_comm_symbols > 0:
                start_idx = self.agent1.puzzle_symbols
                end_idx = start_idx + current_comm_symbols
                
                self.agent2.communication_embedding.weight[start_idx:end_idx].copy_(
                    self.agent1.communication_embedding.weight[start_idx:end_idx]
                )
                comm_params = current_comm_symbols * self.agent1.communication_embedding.embedding_dim
                total_synced_params += comm_params
                print(f"  ✓ Communication embeddings synchronized ({current_comm_symbols} symbols, {comm_params:,} parameters)")
            
            # Update sync tracking
            self.last_sync_cycle = self.cycle_count
            
            print(f"\nSynchronization complete!")
            print(f"Total synchronized parameters: {total_synced_params:,}")
            print(f"Next synchronization in {self.sync_frequency} cycles")
            print(f"{'='*60}")
    
    def should_expand_vocabulary(self) -> bool:
        """Check if it's time to expand the vocabulary"""
        return (self.cycle_count > 0 and 
                self.cycle_count % self.expansion_frequency == 0 and
                self._can_expand())
    
    def _can_expand(self) -> bool:
        """Check if agents can still expand their vocabularies"""
        agent1_can_expand = (
            self.agent1.current_comm_symbols < self.agent1.max_num_symbols - self.agent1.puzzle_symbols or
            self.agent1.current_seq_length < self.agent1.max_seq_length
        )
        agent2_can_expand = (
            self.agent2.current_comm_symbols < self.agent2.max_num_symbols - self.agent2.puzzle_symbols or
            self.agent2.current_seq_length < self.agent2.max_seq_length
        )
        return agent1_can_expand and agent2_can_expand
    
    def expand_vocabularies(self):
        """Expand vocabularies for both agents"""
        print(f"\n{'='*60}")
        print(f"EXPANDING VOCABULARIES AT CYCLE {self.cycle_count}")
        print(f"{'='*60}")
        
        # Show current state
        print("\nBefore expansion:")
        self.agent1.print_position_symbol_mapping()
        self.agent2.print_position_symbol_mapping()
        
        # Expand both agents
        self.agent1.expand_vocabulary(
            additional_symbols=self.symbols_per_expansion,
            additional_length=self.length_per_expansion
        )
        self.agent2.expand_vocabulary(
            additional_symbols=self.symbols_per_expansion,
            additional_length=self.length_per_expansion
        )
        
        # Show new state
        print("\nAfter expansion:")
        self.agent1.print_position_symbol_mapping()
        self.agent2.print_position_symbol_mapping()
        
        # Synchronize communication embeddings for new symbols
        self._synchronize_new_embeddings()
        
        print(f"{'='*60}")
    
    def _synchronize_new_embeddings(self):
        """
        Synchronize communication embeddings between agents for newly added symbols.
        This ensures both agents start with similar representations for new symbols.
        """
        with torch.no_grad():
            # Get the range of newly added symbols
            old_total = self.agent1.current_total_symbols - self.symbols_per_expansion
            new_total = self.agent1.current_total_symbols
            
            if old_total < new_total:
                # Copy new embeddings from agent1 to agent2
                new_embeddings = self.agent1.communication_embedding.weight[old_total:new_total].clone()
                self.agent2.communication_embedding.weight[old_total:new_total] = new_embeddings
                
                print(f"Synchronized embeddings for symbols {old_total} to {new_total-1}")
    
    def set_training_mode(self, mode: str):
        """Set the training mode to control which components are trainable."""
        valid_modes = ["joint", "encoder_only", "decoder_only", "communication_only"]
        if mode not in valid_modes:
            raise ValueError(f"Training mode must be one of {valid_modes}")
        
        self.training_mode = mode
        
        # Set requires_grad for Agent 1
        self._set_component_trainable(self.agent1, "encoder", mode in ["joint", "encoder_only"])
        self._set_component_trainable(self.agent1, "decoder", mode in ["joint", "decoder_only"])
        self._set_component_trainable(self.agent1, "embedding_system", mode in ["joint"])
        self._set_component_trainable(self.agent1, "communication_embedding", mode in ["joint", "communication_only"])
        
        # Set requires_grad for Agent 2
        self._set_component_trainable(self.agent2, "encoder", mode in ["joint", "encoder_only"])
        self._set_component_trainable(self.agent2, "decoder", mode in ["joint", "decoder_only"])
        self._set_component_trainable(self.agent2, "embedding_system", mode in ["joint"])
        self._set_component_trainable(self.agent2, "communication_embedding", mode in ["joint", "communication_only"])
        
        print(f"Set training mode to: {mode}")
    
    def _set_component_trainable(self, agent, component_name, trainable):
        """Helper method to set requires_grad for a component's parameters"""
        if hasattr(agent, component_name):
            component = getattr(agent, component_name)
            for param in component.parameters():
                param.requires_grad = trainable

    def get_message_length_stats(self, symbols: torch.Tensor) -> Dict[str, int]:
        """Calculate message length statistics."""
        nonzero_mask = symbols != 0
        return {
            'total_length': symbols.size(-1),
            'nonzero_symbols': nonzero_mask.sum().item()
        }

    def compute_accuracy(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute accuracy by directly comparing predicted vs actual values."""
        # If reconstructed contains logits, get the predicted indices
        if len(reconstructed.shape) > len(original.shape):
            predictions = reconstructed.argmax(dim=-1)
        else:
            predictions = reconstructed
            
        # Ensure both tensors are of type long for comparison
        original = original.long()
        predictions = predictions.long()
        
        # Calculate accuracy as number of matching positions divided by total positions
        matches = (original == predictions).float()
        accuracy = matches.mean()
        
        return accuracy
    

    def compute_accuracy_with_size(self, target: torch.Tensor, predicted: torch.Tensor) -> dict:
        """Compute both content accuracy on overlapping region and distance-based size accuracy."""
        
        # Get target and predicted dimensions
        target_height, target_width = target.shape[1], target.shape[2]
        pred_height, pred_width = predicted.shape[1], predicted.shape[2]
        
        # Calculate size distance metrics
        height_diff = abs(pred_height - target_height)
        width_diff = abs(pred_width - target_width)
        manhattan_distance = height_diff + width_diff
        
        # Distance-based size accuracy (ranges from 0 to 1)
        # Perfect match = 1.0, gets smaller as distance increases
        size_accuracy_distance = 1.0 / (1.0 + manhattan_distance)
        
        # Alternative: normalized accuracy (optional, you can choose which to use)
        max_possible_distance = target_height + target_width  # Reasonable upper bound
        size_accuracy_normalized = max(0.0, 1.0 - manhattan_distance / max_possible_distance)
        
        # Binary accuracy for comparison
        size_accuracy_binary = float(manhattan_distance == 0)
        
        # Get overlapping region for content accuracy
        min_height = min(predicted.shape[1], target.shape[1])
        min_width = min(predicted.shape[2], target.shape[2])
        
        # For predicted tensor that has an extra dimension for logits
        if len(predicted.shape) > len(target.shape):
            pred_content = predicted[:, :min_height, :min_width].argmax(dim=-1)
        else:
            pred_content = predicted[:, :min_height, :min_width]
        
        target_content = target[:, :min_height, :min_width]
        
        # Compute content accuracy on overlapping region
        matches = (pred_content == target_content).float()
        content_accuracy = matches.mean()
        
        return {
            'size_accuracy': size_accuracy_distance,  # Main metric: distance-based (0-1)
            'size_accuracy_binary': size_accuracy_binary,  # Binary for comparison
            'size_accuracy_normalized': size_accuracy_normalized,  # Alternative metric
            'content_accuracy': content_accuracy.item(),
            'overlap_size': f"{min_height}x{min_width}",
            'size_error': manhattan_distance,  # Raw error for debugging
            'target_size': f"{target_height}x{target_width}",
            'predicted_size': f"{pred_height}x{pred_width}"
        }
    
    def compute_agent_embedding_similarity(self) -> Tuple[float, float, float]:
        """Compute cosine similarity between the embedding tables of both agents."""
        # Get the embedding weights for both agents
        emb1 = self.agent1.embedding_system.symbol_embedding.weight
        emb2 = self.agent2.embedding_system.symbol_embedding.weight
        
        # Split into puzzle and communication symbols
        puzzle_emb1 = emb1[:self.agent1.puzzle_symbols]
        puzzle_emb2 = emb2[:self.agent2.puzzle_symbols]
        
        comm_emb1 = emb1[self.agent1.puzzle_symbols:self.agent1.current_total_symbols]
        comm_emb2 = emb2[self.agent2.puzzle_symbols:self.agent2.current_total_symbols]
        
        # Compute similarities for puzzle symbols
        puzzle_similarities = nn.functional.cosine_similarity(
            puzzle_emb1.unsqueeze(1),
            puzzle_emb2.unsqueeze(0),
            dim=2
        )
        puzzle_sim = torch.diagonal(puzzle_similarities).mean().item()
        
        # Compute similarities for communication symbols (current vocabulary only)
        if comm_emb1.size(0) > 0 and comm_emb2.size(0) > 0:
            comm_similarities = nn.functional.cosine_similarity(
                comm_emb1.unsqueeze(1),
                comm_emb2.unsqueeze(0),
                dim=2
            )
            comm_sim = torch.diagonal(comm_similarities).mean().item()
        else:
            comm_sim = 0.0
        
        # Compute overall similarity across active symbols
        active_emb1 = emb1[:self.agent1.current_total_symbols]
        active_emb2 = emb2[:self.agent2.current_total_symbols]
        all_similarities = nn.functional.cosine_similarity(
            active_emb1.unsqueeze(1),
            active_emb2.unsqueeze(0),
            dim=2
        )
        overall_sim = torch.diagonal(all_similarities).mean().item()
        
        return puzzle_sim, comm_sim, overall_sim

    def train_bidirectional_step(
        self,
        puzzle: torch.Tensor,
        num_exchanges: int = 5,
        temperature: float = 1.0,
        initial_phase: bool = False
    ) -> List[Dict[str, float]]:
        # Check if it's time to synchronize agents
        if self.should_synchronize():
            self.synchronize_agents()
        
        # Check if it's time to expand vocabulary
        if self.should_expand_vocabulary():
            self.expand_vocabularies()
        
        # Increment cycle count
        self.cycle_count += 1
        
        metrics_history = []
        
        # Set components to train/eval mode based on training_mode
        if self.training_mode == "encoder_only":
            self.agent1.encoder.train()
            self.agent1.decoder.eval()
            self.agent2.encoder.train()
            self.agent2.decoder.eval()
        elif self.training_mode == "decoder_only":
            self.agent1.encoder.eval()
            self.agent1.decoder.train()
            self.agent2.encoder.eval()
            self.agent2.decoder.train()
            self.agent1.communication_embedding.eval()
            self.agent2.communication_embedding.eval()
        elif self.training_mode == "communication_only":
            self.agent1.encoder.eval()
            self.agent1.decoder.eval()
            self.agent2.encoder.eval()
            self.agent2.decoder.eval()
            self.agent1.communication_embedding.train()
            self.agent2.communication_embedding.train()
        else:  # joint training
            self.agent1.train()
            self.agent2.train()

        def get_module_gradients(module):
            """Calculate average gradient magnitude for a module's parameters"""
            total_grad = 0.0
            num_params = 0
            for name, param in module.named_parameters():
                if param.grad is not None and param.requires_grad:
                    total_grad += param.grad.abs().mean().item()
                    num_params += 1
            return total_grad / num_params if num_params > 0 else 0.0

        def print_gradient_info(sender, receiver):
            """Print gradient information for encoder and decoder components"""
            print("\nGradient Magnitudes:")
            
            # Sender gradients
            print("  Sender:")
            print(f"    Encoder: {get_module_gradients(sender.encoder):.6f} (trainable: {next(sender.encoder.parameters()).requires_grad})")
            print(f"    Decoder: {get_module_gradients(sender.decoder):.6f} (trainable: {next(sender.decoder.parameters()).requires_grad})")
            print(f"    Embedding System: {get_module_gradients(sender.embedding_system):.6f} (trainable: {next(sender.embedding_system.parameters()).requires_grad})")
            print(f"    Communication Embedding: {get_module_gradients(sender.communication_embedding):.6f} (trainable: {next(sender.communication_embedding.parameters()).requires_grad})")
            
            # Receiver gradients
            print("  Receiver:")
            print(f"    Encoder: {get_module_gradients(receiver.encoder):.6f} (trainable: {next(receiver.encoder.parameters()).requires_grad})")
            print(f"    Decoder: {get_module_gradients(receiver.decoder):.6f} (trainable: {next(receiver.decoder.parameters()).requires_grad})")
            print(f"    Embedding System: {get_module_gradients(receiver.embedding_system):.6f} (trainable: {next(receiver.embedding_system.parameters()).requires_grad})")
            print(f"    Communication Embedding: {get_module_gradients(receiver.communication_embedding):.6f} (trainable: {next(receiver.communication_embedding.parameters()).requires_grad})")
        
        for exchange in range(num_exchanges):
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            
            # Agent1 encodes, Agent2 decodes
            symbols1, _, length_stats1 = self.agent1.encode_puzzle_to_message(
                puzzle, temperature=temperature, initial_phase=initial_phase
            )
            
            reconstructed1, grid_logits1, intermediates1, confidences1, size_logits1 = self.agent2.decode_message_to_puzzle(
                symbols1, temperature=temperature, hard=True 
            )
            
            # Agent2 encodes, Agent1 decodes
            symbols2, _, length_stats2 = self.agent2.encode_puzzle_to_message(
                puzzle, temperature=temperature, initial_phase=initial_phase
            )
            
            reconstructed2, grid_logits2, intermediates2, confidences2, size_logits2 = self.agent1.decode_message_to_puzzle(
                symbols2, temperature=temperature, hard=True 
            )
            
            # Compute grid losses
            grid_loss1 = self.compute_grid_loss_with_padding(grid_logits1, puzzle)
            grid_loss2 = self.compute_grid_loss_with_padding(grid_logits2, puzzle)
            
            # Calculate size losses
            target_size = (puzzle.size(1), puzzle.size(2))
            height_target = torch.tensor(target_size[0] - 1, device=grid_loss1.device)
            width_target = torch.tensor(target_size[1] - 1, device=grid_loss1.device)
            
            # Compute size losses
            size_loss1_height = F.cross_entropy(size_logits1[0], height_target.unsqueeze(0))
            size_loss1_width = F.cross_entropy(size_logits1[1], width_target.unsqueeze(0))
            size_loss2_height = F.cross_entropy(size_logits2[0], height_target.unsqueeze(0))
            size_loss2_width = F.cross_entropy(size_logits2[1], width_target.unsqueeze(0))
            
            size_loss1 = size_loss1_height + size_loss1_width
            size_loss2 = size_loss2_height + size_loss2_width
            
            # Add size accuracy penalty
            pred_height1 = size_logits1[0].argmax(dim=-1) + 1
            pred_width1 = size_logits1[1].argmax(dim=-1) + 1
            pred_height2 = size_logits2[0].argmax(dim=-1) + 1
            pred_width2 = size_logits2[1].argmax(dim=-1) + 1
            
            size_penalty1 = torch.abs(pred_height1 - target_size[0]) + torch.abs(pred_width1 - target_size[1])
            size_penalty2 = torch.abs(pred_height2 - target_size[0]) + torch.abs(pred_width2 - target_size[1])
            size_penalty = (size_penalty1 + size_penalty2).float() * 0.1
            
            # Compute total loss with increased size emphasis
            size_weight = 1.0
            total_loss = (
                grid_loss1 + grid_loss2 +
                size_weight * (size_loss1 + size_loss2) 
                # + size_weight * size_penalty
            )
            
            # Add symbol entropy regularization when training encoders
            if self.training_mode in ["encoder_only", "joint"]:
                entropy1 = self._calculate_symbol_entropy(symbols1)
                entropy2 = self._calculate_symbol_entropy(symbols2)
                
                entropy_weight = 0.1
                total_loss = total_loss - entropy_weight * (entropy1 + entropy2)
            
            total_loss.backward()
            
            # Print gradient information
            # print_gradient_info(self.agent1, self.agent2)
            
            self.opt1.step()
            self.opt2.step()
            
            # Compute accuracies
            with torch.no_grad():
                accuracy1 = self.compute_accuracy_with_size(puzzle, reconstructed1)
                accuracy2 = self.compute_accuracy_with_size(puzzle, reconstructed2)
            
            metrics = {
                'cycle': self.cycle_count,
                'total_loss': total_loss.item(),
                'grid_loss1': grid_loss1.item(),
                'grid_loss2': grid_loss2.item(),
                'size_loss1': size_loss1.item(),
                'size_loss2': size_loss2.item(),
                'size_penalty': size_penalty.item(),
                
                # Updated size accuracy metrics
                'agent1_size_accuracy': accuracy1['size_accuracy'],  # Distance-based
                'agent1_size_binary': accuracy1['size_accuracy_binary'],  # Binary for comparison
                'agent1_size_error': accuracy1['size_error'],  # Raw error
                'agent1_content_accuracy': accuracy1['content_accuracy'],
                'agent1_overlap': accuracy1['overlap_size'],
                'agent1_target_size': accuracy1['target_size'],
                'agent1_predicted_size': accuracy1['predicted_size'],
                
                'agent2_size_accuracy': accuracy2['size_accuracy'],  # Distance-based
                'agent2_size_binary': accuracy2['size_accuracy_binary'],  # Binary for comparison
                'agent2_size_error': accuracy2['size_error'],  # Raw error
                'agent2_content_accuracy': accuracy2['content_accuracy'],
                'agent2_overlap': accuracy2['overlap_size'],
                'agent2_target_size': accuracy2['target_size'],
                'agent2_predicted_size': accuracy2['predicted_size'],
                
                'message_length1': length_stats1['total_length'],
                'message_length2': length_stats2['total_length'],
                'nonzero_symbols1': length_stats1['nonzero_symbols'],
                'nonzero_symbols2': length_stats2['nonzero_symbols'],
                'training_mode': self.training_mode,
                'vocab_size1': length_stats1['current_vocab_size'],
                'vocab_size2': length_stats2['current_vocab_size'],
                'seq_length1': length_stats1['current_seq_length'],
                'seq_length2': length_stats2['current_seq_length']
            }
            
            # If training encoders, add symbol distribution metrics
            if self.training_mode in ["encoder_only", "joint"]:
                _, max_symbol1 = symbols1.max(dim=-1)
                _, max_symbol2 = symbols2.max(dim=-1)
                
                metrics['encoder1_symbol'] = max_symbol1[0, 0].item()
                metrics['encoder2_symbol'] = max_symbol2[0, 0].item()
            
            metrics_history.append(metrics)
            
        return metrics_history
    
    def _calculate_symbol_entropy(self, symbol_probs):
        """Calculate the entropy of symbol probability distributions."""
        eps = 1e-8
        probs = symbol_probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean()

    def _update_learning_rates(self, current_lr: float) -> None:
        """Update learning rates for both optimizers."""
        if not self.in_initial_phase:
            for param_group in self.opt1.param_groups:
                param_group['lr'] = current_lr
        for param_group in self.opt2.param_groups:
            param_group['lr'] = current_lr

    def compute_grid_loss(self, grid_logits: torch.Tensor, target: torch.Tensor, num_symbols: int) -> torch.Tensor:
        logits = grid_logits.view(-1, num_symbols)
        target_flat = target.view(-1)
        
        target_flat = torch.clamp(target_flat, min=0, max=num_symbols-1)
        
        ce_loss = F.cross_entropy(
            logits, 
            target_flat, 
            label_smoothing=0.0,
            reduction='mean'
        )
        
        return ce_loss

    def compute_grid_loss_with_padding(self, grid_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss between predicted and target grids with size mismatch penalty."""
        pred_height, pred_width = grid_logits.shape[1:3]
        target_height, target_width = target.shape[1:3]
        
        size_penalty = torch.tensor(
            abs(pred_height - target_height) + abs(pred_width - target_width),
            device=grid_logits.device,
            dtype=torch.float32
        )
        
        min_height = min(pred_height, target_height)
        min_width = min(pred_width, target_width)
        
        pred_region = grid_logits[:, :min_height, :min_width, :]
        target_region = target[:, :min_height, :min_width]
        
        content_loss = F.cross_entropy(
            pred_region.reshape(-1, grid_logits.size(-1)),
            target_region.reshape(-1),
            reduction='mean'
        )
        
        total_loss = content_loss + 0.1 * size_penalty
        
        return total_loss.clamp(min=0.0, max=.0)

    def get_vocabulary_status(self) -> Dict[str, Dict[str, int]]:
        """Get current vocabulary status for both agents"""
        return {
            'agent1': self.agent1.get_vocabulary_info(),
            'agent2': self.agent2.get_vocabulary_info(),
            'cycle_count': self.cycle_count,
            'next_expansion_at': ((self.cycle_count // self.expansion_frequency) + 1) * self.expansion_frequency,
            'next_sync_at': self.last_sync_cycle + self.sync_frequency
        }


# Create a factory function to replace the original CommunicationTrainer
def CommunicationTrainer(*args, **kwargs):
    """Factory function to create ProgressiveCommunicationTrainer instances"""
    return ProgressiveCommunicationTrainer(*args, **kwargs)