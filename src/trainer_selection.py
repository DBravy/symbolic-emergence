import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from agent_selection import ProgressiveSelectionAgent
from puzzle import Puzzle
import numpy as np
import random

class ProgressiveSelectionTrainer:
    def __init__(
        self,
        agent1: ProgressiveSelectionAgent,
        agent2: ProgressiveSelectionAgent,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        expansion_frequency: int = 200,  # Expand every N cycles
        symbols_per_expansion: int = 2,   # Add N symbols per expansion
        length_per_expansion: int = 1,    # Add N sequence length per expansion
        sync_frequency: int = 50,         # Synchronize every N cycles
        num_distractors: int = 3,         # Number of distractor puzzles
        distractor_strategy: str = 'random'  # 'random', 'similar_size', 'hard'
    ):
        self.agent1 = agent1.to(device)
        self.agent2 = agent2.to(device)
        self.device = device
        
        # Progressive training parameters
        self.expansion_frequency = expansion_frequency
        self.symbols_per_expansion = symbols_per_expansion
        self.length_per_expansion = length_per_expansion
        self.cycle_count = 0
        
        # Synchronization parameters
        self.sync_frequency = sync_frequency
        self.last_sync_cycle = 0

        # Selection task parameters
        self.num_distractors = num_distractors
        self.distractor_strategy = distractor_strategy

        # Copy Agent 1's puzzle symbol embeddings to Agent 2
        with torch.no_grad():
            puzzle_embeddings = agent1.embedding_system.symbol_embedding.weight[:agent1.puzzle_symbols].clone()
            agent2.embedding_system.symbol_embedding.weight[:agent2.puzzle_symbols] = puzzle_embeddings

        # Initialize optimizers - include similarity components
        self.opt1 = optim.Adam([
            {'params': agent1.embedding_system.parameters(), 'lr': learning_rate},
            {'params': agent1.encoder.parameters(), 'lr': learning_rate},
            {'params': agent1.communication_embedding.parameters(), 'lr': learning_rate},
            {'params': agent1.message_pooling.parameters(), 'lr': learning_rate}
        ])

        # Add similarity MLP to optimizer if it exists
        if hasattr(agent1, 'similarity_mlp'):
            self.opt1.add_param_group({'params': agent1.similarity_mlp.parameters(), 'lr': learning_rate})

        self.opt2 = optim.Adam([
            {'params': agent2.embedding_system.parameters(), 'lr': learning_rate},
            {'params': agent2.encoder.parameters(), 'lr': learning_rate},
            {'params': agent2.communication_embedding.parameters(), 'lr': learning_rate},
            {'params': agent2.message_pooling.parameters(), 'lr': learning_rate}
        ])

        # Add similarity MLP to optimizer if it exists
        if hasattr(agent2, 'similarity_mlp'):
            self.opt2.add_param_group({'params': agent2.similarity_mlp.parameters(), 'lr': learning_rate})

        self.in_initial_phase = True
        
        # Loss criteria
        self.selection_criterion = nn.CrossEntropyLoss()
        
        # Training mode tracking
        self.training_mode = "joint"
        
        # Store puzzle dataset for distractor sampling
        self.puzzle_dataset = []
    
    def set_puzzle_dataset(self, puzzles: List[Puzzle]):
        """Set the puzzle dataset for distractor sampling"""
        self.puzzle_dataset = puzzles
        print(f"Loaded {len(puzzles)} puzzles for distractor sampling")
    
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
        """Synchronize ALL agent parameters from agent1 to agent2."""
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
            
            # 3. Synchronize message pooling
            print("Synchronizing message pooling...")
            pooling_params = self._sync_module_recursive(
                self.agent1.message_pooling,
                self.agent2.message_pooling,
                "message_pooling"
            )
            total_synced_params += pooling_params
            print(f"  ✓ Message pooling synchronized ({pooling_params:,} parameters)")
            
            # 4. Synchronize similarity MLP if it exists
            if hasattr(self.agent1, 'similarity_mlp') and hasattr(self.agent2, 'similarity_mlp'):
                print("Synchronizing similarity MLP...")
                similarity_params = self._sync_module_recursive(
                    self.agent1.similarity_mlp,
                    self.agent2.similarity_mlp,
                    "similarity_mlp"
                )
                total_synced_params += similarity_params
                print(f"  ✓ Similarity MLP synchronized ({similarity_params:,} parameters)")
            
            # 5. Synchronize communication embeddings (current vocabulary only)
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
        """Synchronize communication embeddings between agents for newly added symbols."""
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
        valid_modes = ["joint", "encoder_only", "selection_only"]
        if mode not in valid_modes:
            raise ValueError(f"Training mode must be one of {valid_modes}")
        
        self.training_mode = mode
        
        # Set requires_grad for Agent 1
        self._set_component_trainable(self.agent1, "encoder", mode in ["joint", "encoder_only"])
        self._set_component_trainable(self.agent1, "embedding_system", mode in ["joint"])
        self._set_component_trainable(self.agent1, "communication_embedding", mode in ["joint"])
        self._set_component_trainable(self.agent1, "message_pooling", mode in ["joint", "selection_only"])
        
        if hasattr(self.agent1, 'similarity_mlp'):
            self._set_component_trainable(self.agent1, "similarity_mlp", mode in ["joint", "selection_only"])
        
        # Set requires_grad for Agent 2
        self._set_component_trainable(self.agent2, "encoder", mode in ["joint", "encoder_only"])
        self._set_component_trainable(self.agent2, "embedding_system", mode in ["joint"])
        self._set_component_trainable(self.agent2, "communication_embedding", mode in ["joint"])
        self._set_component_trainable(self.agent2, "message_pooling", mode in ["joint", "selection_only"])
        
        if hasattr(self.agent2, 'similarity_mlp'):
            self._set_component_trainable(self.agent2, "similarity_mlp", mode in ["joint", "selection_only"])
        
        print(f"Set training mode to: {mode}")
    
    def _set_component_trainable(self, agent, component_name, trainable):
        """Helper method to set requires_grad for a component's parameters"""
        if hasattr(agent, component_name):
            component = getattr(agent, component_name)
            for param in component.parameters():
                param.requires_grad = trainable

    def sample_distractors(self, target_puzzle: torch.Tensor, target_idx: int) -> List[torch.Tensor]:
        """
        Sample distractor puzzles based on the specified strategy.
        
        Args:
            target_puzzle: [batch_size, height, width] - target puzzle
            target_idx: index of target puzzle in dataset
            
        Returns:
            List of distractor puzzle tensors
        """
        if len(self.puzzle_dataset) < self.num_distractors + 1:
            raise ValueError(f"Need at least {self.num_distractors + 1} puzzles for selection task")
        
        distractors = []
        available_indices = list(range(len(self.puzzle_dataset)))
        available_indices.remove(target_idx)  # Don't sample the target
        
        if self.distractor_strategy == 'random':
            # Simple random sampling
            distractor_indices = random.sample(available_indices, self.num_distractors)
            
        elif self.distractor_strategy == 'similar_size':
            # Sample puzzles with similar dimensions
            target_height, target_width = target_puzzle.shape[1], target_puzzle.shape[2]
            
            # Calculate size differences
            size_diffs = []
            for idx in available_indices:
                puzzle = self.puzzle_dataset[idx]
                puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long)
                h_diff = abs(puzzle_tensor.shape[0] - target_height)
                w_diff = abs(puzzle_tensor.shape[1] - target_width)
                size_diff = h_diff + w_diff
                size_diffs.append((size_diff, idx))
            
            # Sort by size similarity and take closest ones
            size_diffs.sort(key=lambda x: x[0])
            distractor_indices = [idx for _, idx in size_diffs[:self.num_distractors]]
            
        elif self.distractor_strategy == 'hard':
            # TODO: Implement hard negative sampling based on embedding similarity
            # For now, fall back to random
            distractor_indices = random.sample(available_indices, self.num_distractors)
            
        else:
            raise ValueError(f"Unknown distractor strategy: {self.distractor_strategy}")
        
        # Convert to tensors
        for idx in distractor_indices:
            puzzle = self.puzzle_dataset[idx]
            distractor_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0)  # Add batch dimension
            distractors.append(distractor_tensor)
        
        return distractors

    def train_bidirectional_step(
        self,
        puzzle: torch.Tensor,
        puzzle_idx: int,  # NEW: Index of puzzle in dataset for distractor sampling
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
            self.agent1.embedding_system.train()
            self.agent1.message_pooling.eval()
            if hasattr(self.agent1, 'similarity_mlp'):
                self.agent1.similarity_mlp.eval()
                
            self.agent2.encoder.train()
            self.agent2.embedding_system.train()
            self.agent2.message_pooling.eval()
            if hasattr(self.agent2, 'similarity_mlp'):
                self.agent2.similarity_mlp.eval()
                
        elif self.training_mode == "selection_only":
            self.agent1.encoder.eval()
            self.agent1.embedding_system.eval()
            self.agent1.message_pooling.train()
            if hasattr(self.agent1, 'similarity_mlp'):
                self.agent1.similarity_mlp.train()
                
            self.agent2.encoder.eval()
            self.agent2.embedding_system.eval()
            self.agent2.message_pooling.train()
            if hasattr(self.agent2, 'similarity_mlp'):
                self.agent2.similarity_mlp.train()
                
        else:  # joint training
            self.agent1.train()
            self.agent2.train()
        
        for exchange in range(num_exchanges):
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            
            # Sample distractors for this exchange
            distractors = self.sample_distractors(puzzle, puzzle_idx)
            
            # Agent1 encodes, Agent2 selects
            symbols1, symbol_logits1, length_stats1 = self.agent1.encode_puzzle_to_message(
                puzzle, temperature=temperature, initial_phase=initial_phase
            )
            
            # Prepare candidates: target + distractors
            candidates1 = [puzzle] + distractors
            
            # Receiver selects from candidates
            selection_probs1, selection_logits1, debug_info1 = self.agent2.select_from_candidates(
                symbols1, candidates1, temperature=temperature
            )
            
            # Agent2 encodes, Agent1 selects (bidirectional)
            symbols2, symbol_logits2, length_stats2 = self.agent2.encode_puzzle_to_message(
                puzzle, temperature=temperature, initial_phase=initial_phase
            )
            
            # Sample different distractors for reverse direction
            distractors2 = self.sample_distractors(puzzle, puzzle_idx)
            candidates2 = [puzzle] + distractors2
            
            selection_probs2, selection_logits2, debug_info2 = self.agent1.select_from_candidates(
                symbols2, candidates2, temperature=temperature
            )
            
            # Compute selection losses (target is always at index 0)
            target_idx = torch.tensor([0], device=self.device, dtype=torch.long)
            
            selection_loss1 = self.selection_criterion(
                selection_logits1, target_idx.expand(selection_logits1.size(0))
            )
            selection_loss2 = self.selection_criterion(
                selection_logits2, target_idx.expand(selection_logits2.size(0))
            )
            
            # Total loss
            total_loss = selection_loss1 + selection_loss2
            
            # Add symbol entropy regularization when training encoders
            if self.training_mode in ["encoder_only", "joint"]:
                entropy1 = self._calculate_symbol_entropy(symbols1)
                entropy2 = self._calculate_symbol_entropy(symbols2)
                
                entropy_weight = 0.1
                total_loss = total_loss - entropy_weight * (entropy1 + entropy2)
            
            total_loss.backward()
            
            self.opt1.step()
            self.opt2.step()
            
            # Compute accuracies
            with torch.no_grad():
                # Selection accuracy (did the agent choose the correct puzzle?)
                pred1 = selection_logits1.argmax(dim=-1)
                pred2 = selection_logits2.argmax(dim=-1)
                
                acc1 = (pred1 == target_idx[0]).float().mean().item()
                acc2 = (pred2 == target_idx[0]).float().mean().item()
                
                # Confidence in correct selection
                correct_confidence1 = selection_probs1[0, 0].item()  # Probability of target
                correct_confidence2 = selection_probs2[0, 0].item()
            
            metrics = {
                'cycle': self.cycle_count,
                'total_loss': total_loss.item(),
                'selection_loss1': selection_loss1.item(),
                'selection_loss2': selection_loss2.item(),
                
                # Selection accuracies
                'agent1_selection_accuracy': acc1,
                'agent2_selection_accuracy': acc2,
                
                # Confidence in correct answer
                'agent1_correct_confidence': correct_confidence1,
                'agent2_correct_confidence': correct_confidence2,
                
                # Message statistics
                'message_length1': length_stats1['total_length'],
                'message_length2': length_stats2['total_length'],
                'nonzero_symbols1': length_stats1['nonzero_symbols'],
                'nonzero_symbols2': length_stats2['nonzero_symbols'],
                'training_mode': self.training_mode,
                'vocab_size1': length_stats1['current_vocab_size'],
                'vocab_size2': length_stats2['current_vocab_size'],
                'seq_length1': length_stats1['current_seq_length'],
                'seq_length2': length_stats2['current_seq_length'],
                
                # Selection task specific
                'num_candidates': len(candidates1),
                'distractor_strategy': self.distractor_strategy,
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

    def get_vocabulary_status(self) -> Dict[str, Dict[str, int]]:
        """Get current vocabulary status for both agents"""
        return {
            'agent1': self.agent1.get_vocabulary_info(),
            'agent2': self.agent2.get_vocabulary_info(),
            'cycle_count': self.cycle_count,
            'next_expansion_at': ((self.cycle_count // self.expansion_frequency) + 1) * self.expansion_frequency,
            'next_sync_at': self.last_sync_cycle + self.sync_frequency,
            'selection_config': {
                'num_distractors': self.num_distractors,
                'distractor_strategy': self.distractor_strategy
            }
        }


# Create a factory function to replace the original CommunicationTrainer
def CommunicationTrainer(*args, **kwargs):
    """Factory function to create ProgressiveSelectionTrainer instances"""
    return ProgressiveSelectionTrainer(*args, **kwargs)