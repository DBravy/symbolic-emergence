import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

class ContrastiveEncoderTrainer:
    """
    Implements contrastive pre-training for the communication system's encoders.
    This helps establish consistent symbol assignments before joint training.
    """
    
    def __init__(self, trainer, temperature=0.1):
        """
        Initialize the contrastive trainer with a reference to the main trainer.
        
        Args:
            trainer: The CommunicationTrainer instance
            temperature: Temperature parameter for contrastive loss scaling
        """
        self.trainer = trainer
        self.temperature = temperature
        self.device = trainer.device
        self.agent1 = trainer.agent1
        self.agent2 = trainer.agent2
        
        # Use trainer's optimizers, but only train encoder components
        self.set_contrastive_training_mode()
        
    def set_contrastive_training_mode(self):
        """Set only encoder components to trainable mode"""
        # Define components to train during contrastive learning
        for agent in [self.agent1, self.agent2]:
            # Train encoder and embedding system
            for param in agent.encoder.parameters():
                param.requires_grad = True
            for param in agent.embedding_system.parameters():
                param.requires_grad = True
                
            # Freeze decoder and communication embedding
            for param in agent.decoder.parameters():
                param.requires_grad = False
            for param in agent.communication_embedding.parameters():
                param.requires_grad = False
    
    def generate_positive_pairs(self, puzzle_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate positive pairs (similar puzzle variations) for contrastive learning.
        
        Args:
            puzzle_tensor: Original puzzle tensor [B, H, W]
            
        Returns:
            List of transformed puzzle tensors
        """
        transformed = []
        
        # 90-degree rotation
        rotated = torch.rot90(puzzle_tensor, k=1, dims=[1, 2])
        transformed.append(rotated)
        
        # Horizontal flip
        flipped = torch.flip(puzzle_tensor, dims=[2])
        transformed.append(flipped)
        
        # Small noise (random pixel changes)
        noisy = puzzle_tensor.clone()
        mask = torch.rand_like(noisy.float()) < 0.1  # Change 10% of pixels
        noise = torch.randint_like(noisy, 0, self.agent1.puzzle_symbols)
        noisy = torch.where(mask, noise, noisy)
        transformed.append(noisy)
        
        return transformed
    
    def compute_contrastive_loss(self, 
                               anchor_symbols: torch.Tensor, 
                               positive_symbols: List[torch.Tensor],
                               negative_symbols: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss to push similar puzzle encodings together
        and different puzzle encodings apart.
        
        Args:
            anchor_symbols: Symbol distribution from anchor puzzle [B, seq_len, num_symbols]
            positive_symbols: List of symbol distributions from similar puzzles
            negative_symbols: List of symbol distributions from different puzzles
            
        Returns:
            Contrastive loss value
        """
        # For simplicity, use only the first token position since max_seq_length = 1
        anchor_flat = anchor_symbols[:, 0, :]  # [B, num_symbols]
        
        # Normalize vectors for cosine similarity
        anchor_norm = F.normalize(anchor_flat, p=2, dim=1)
        
        # Process positive samples
        pos_similarities = []
        for pos_symbols in positive_symbols:
            pos_flat = pos_symbols[:, 0, :]
            pos_norm = F.normalize(pos_flat, p=2, dim=1)
            similarity = F.cosine_similarity(anchor_norm, pos_norm)
            pos_similarities.append(similarity.unsqueeze(1))
        
        # Stack positive similarities [B, num_positives]
        if pos_similarities:
            pos_similarities = torch.cat(pos_similarities, dim=1) / self.temperature
        else:
            # Handle case with no positive pairs
            pos_similarities = torch.tensor([]).to(self.device)
        
        # Process negative samples
        neg_similarities = []
        for neg_symbols in negative_symbols:
            neg_flat = neg_symbols[:, 0, :]
            neg_norm = F.normalize(neg_flat, p=2, dim=1)
            similarity = F.cosine_similarity(anchor_norm, neg_norm) / self.temperature
            neg_similarities.append(similarity.unsqueeze(1))
        
        # Stack negative similarities [B, num_negatives]
        if neg_similarities:
            neg_similarities = torch.cat(neg_similarities, dim=1)
        else:
            # Handle case with no negative pairs
            neg_similarities = torch.tensor([]).to(self.device)
        
        # Push positives together (maximize similarity)
        positive_loss = 0
        if len(pos_similarities) > 0:
            positive_loss = -torch.mean(pos_similarities)
        
        # Push negatives apart with margin (minimize similarity)
        negative_loss = 0
        if len(neg_similarities) > 0:
            margin = 0.3  # Don't push further than this margin
            negative_loss = torch.clamp(neg_similarities - margin, min=0.0).mean()
        
        # Combine losses (with optional weighting if needed)
        total_loss = positive_loss + negative_loss
        
        return total_loss
    
    def add_entropy_regularization(self, 
                                  symbols: torch.Tensor, 
                                  weight: float = 0.1) -> torch.Tensor:
        """
        Add entropy regularization to encourage deterministic output distributions.
        Lower entropy means more peaked (concentrated) distributions.
        
        Args:
            symbols: Symbol distributions [B, seq_len, num_symbols]
            weight: Weight for entropy term
            
        Returns:
            Entropy regularization loss
        """
        # Calculate entropy of the distribution
        # First token position only (since max_seq_length = 1)
        probs = F.softmax(symbols[:, 0, :], dim=-1)
        log_probs = F.log_softmax(symbols[:, 0, :], dim=-1)
        
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # Return weighted entropy (we want to minimize entropy)
        return weight * entropy
    
    def train_batch(self, 
                   puzzle_batch: List[torch.Tensor], 
                   num_iterations: int = 1) -> Dict:
        """
        Train encoders with contrastive learning on a batch of puzzles.
        
        Args:
            puzzle_batch: List of puzzle tensors [B, H, W]
            num_iterations: Number of training iterations per batch
            
        Returns:
            Dictionary of metrics
        """
        batch_size = len(puzzle_batch)
        if batch_size < 2:
            return {"loss": 0.0, "entropy": 0.0}
        
        total_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(num_iterations):
            self.trainer.opt1.zero_grad()
            self.trainer.opt2.zero_grad()
            
            # Process each puzzle to get encodings
            agent1_encodings = []
            agent2_encodings = []
            
            for puzzle in puzzle_batch:
                # Agent 1 encoding
                symbols1, _, _ = self.agent1.encode_puzzle_to_message(
                    puzzle, temperature=self.temperature
                )
                agent1_encodings.append(symbols1)
                
                # Agent 2 encoding
                symbols2, _, _ = self.agent2.encode_puzzle_to_message(
                    puzzle, temperature=self.temperature
                )
                agent2_encodings.append(symbols2)
            
            # Compute contrastive loss for each puzzle in the batch
            batch_loss = 0.0
            batch_entropy = 0.0
            
            for i in range(batch_size):
                # Generate positive pairs for the current puzzle
                positive_transforms = self.generate_positive_pairs(puzzle_batch[i])
                
                # Get encodings for positive pairs
                agent1_positives = []
                agent2_positives = []
                
                for transform in positive_transforms:
                    # Agent 1
                    pos_symbols1, _, _ = self.agent1.encode_puzzle_to_message(
                        transform, temperature=self.temperature
                    )
                    agent1_positives.append(pos_symbols1)
                    
                    # Agent 2
                    pos_symbols2, _, _ = self.agent2.encode_puzzle_to_message(
                        transform, temperature=self.temperature
                    )
                    agent2_positives.append(pos_symbols2)
                
                # Negative pairs are all other puzzles in the batch
                agent1_negatives = [agent1_encodings[j] for j in range(batch_size) if j != i]
                agent2_negatives = [agent2_encodings[j] for j in range(batch_size) if j != i]
                
                # Compute contrastive losses
                agent1_loss = self.compute_contrastive_loss(
                    agent1_encodings[i], agent1_positives, agent1_negatives
                )
                
                agent2_loss = self.compute_contrastive_loss(
                    agent2_encodings[i], agent2_positives, agent2_negatives
                )
                
                # Add entropy regularization
                agent1_entropy = self.add_entropy_regularization(agent1_encodings[i])
                agent2_entropy = self.add_entropy_regularization(agent2_encodings[i])
                
                # Accumulate losses and entropy
                puzzle_loss = agent1_loss + agent2_loss + agent1_entropy + agent2_entropy
                batch_loss += puzzle_loss
                batch_entropy += agent1_entropy + agent2_entropy
            
            # Average loss across the batch
            batch_loss /= batch_size
            batch_entropy /= batch_size
            
            # Backpropagate
            batch_loss.backward()
            self.trainer.opt1.step()
            self.trainer.opt2.step()
            
            # Accumulate metrics
            total_loss += batch_loss.item()
            total_entropy += batch_entropy.item()
        
        # Average over iterations
        avg_loss = total_loss / num_iterations
        avg_entropy = total_entropy / num_iterations
        
        return {"loss": avg_loss, "entropy": avg_entropy}
    
    def check_encoder_consistency(self, 
                                puzzle_batch: List[torch.Tensor], 
                                num_passes: int = 10) -> Dict:
        """
        Check how consistent encoders are in their symbol assignments.
        
        Args:
            puzzle_batch: List of puzzle tensors
            num_passes: Number of forward passes to compute consistency
            
        Returns:
            Dictionary of consistency metrics
        """
        with torch.no_grad():
            consistency_metrics = {
                'agent1_consistency': [],
                'agent2_consistency': [],
                'symbol_assignments': {}
            }
            
            for puzzle_idx, puzzle in enumerate(puzzle_batch):
                # Multiple forward passes
                agent1_symbols = []
                agent2_symbols = []
                
                for _ in range(num_passes):
                    # Agent 1 encoding
                    symbols1, _, _ = self.agent1.encode_puzzle_to_message(
                        puzzle, temperature=self.temperature
                    )
                    # Get the most likely symbol at first position
                    symbol1 = symbols1[0, 0].argmax().item() 
                    agent1_symbols.append(symbol1)
                    
                    # Agent 2 encoding
                    symbols2, _, _ = self.agent2.encode_puzzle_to_message(
                        puzzle, temperature=self.temperature
                    )
                    symbol2 = symbols2[0, 0].argmax().item()
                    agent2_symbols.append(symbol2)
                
                # Compute consistency as percentage of passes with the same symbol
                from collections import Counter
                counter1 = Counter(agent1_symbols)
                counter2 = Counter(agent2_symbols)
                
                # Most common symbol and its frequency
                most_common1 = counter1.most_common(1)[0]
                most_common2 = counter2.most_common(1)[0]
                
                consistency1 = most_common1[1] / num_passes
                consistency2 = most_common2[1] / num_passes
                
                consistency_metrics['agent1_consistency'].append(consistency1)
                consistency_metrics['agent2_consistency'].append(consistency2)
                
                # Record most common symbol assignment
                consistency_metrics['symbol_assignments'][f'puzzle_{puzzle_idx}'] = {
                    'agent1_symbol': most_common1[0],
                    'agent1_confidence': consistency1,
                    'agent2_symbol': most_common2[0],
                    'agent2_confidence': consistency2
                }
            
            # Compute average consistency across puzzles
            consistency_metrics['avg_agent1_consistency'] = np.mean(consistency_metrics['agent1_consistency'])
            consistency_metrics['avg_agent2_consistency'] = np.mean(consistency_metrics['agent2_consistency'])
            
            return consistency_metrics
    
    def pretrain(self, 
                puzzles: List, 
                num_epochs: int = 500, 
                batch_size: int = 8,
                check_every: int = 50) -> Dict:
        """
        Pretrain encoders using contrastive learning.
        
        Args:
            puzzles: List of puzzles
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            check_every: Check consistency every N epochs
            
        Returns:
            Training history
        """
        num_puzzles = len(puzzles)
        history = {
            'loss': [],
            'entropy': [],
            'agent1_consistency': [],
            'agent2_consistency': [],
            'consistency_checks': []
        }
        
        print(f"Starting contrastive pre-training with {num_puzzles} puzzles for {num_epochs} epochs")
        
        # Convert puzzles to tensors
        puzzle_tensors = [
            torch.tensor(puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
            for puzzle in puzzles
        ]
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_entropy = 0.0
            
            # Shuffle puzzles for each epoch
            indices = torch.randperm(num_puzzles)
            
            # Process in batches
            num_batches = 0
            for i in range(0, num_puzzles, batch_size):
                # Get batch indices
                batch_indices = indices[i:min(i+batch_size, num_puzzles)]
                if len(batch_indices) < 2:  # Need at least 2 puzzles for contrastive learning
                    continue
                
                # Get batch tensors
                batch_tensors = [puzzle_tensors[idx] for idx in batch_indices]
                
                # Train on this batch
                metrics = self.train_batch(batch_tensors, num_iterations=1)
                
                epoch_loss += metrics['loss']
                epoch_entropy += metrics['entropy']
                num_batches += 1
            
            # Compute epoch averages
            if num_batches > 0:
                epoch_loss /= num_batches
                epoch_entropy /= num_batches
            
            # Record history
            history['loss'].append(epoch_loss)
            history['entropy'].append(epoch_entropy)
            
            # Check consistency periodically
            if (epoch + 1) % check_every == 0 or epoch == num_epochs - 1:
                # Select a small subset of puzzles for checking
                check_size = min(10, num_puzzles)
                check_indices = np.random.choice(num_puzzles, check_size, replace=False)
                check_tensors = [puzzle_tensors[idx] for idx in check_indices]
                
                # Check encoder consistency
                consistency = self.check_encoder_consistency(check_tensors)
                
                # Record consistency metrics
                history['agent1_consistency'].append(consistency['avg_agent1_consistency'])
                history['agent2_consistency'].append(consistency['avg_agent2_consistency'])
                history['consistency_checks'].append(consistency)
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={epoch_loss:.4f}, "
                      f"Entropy={epoch_entropy:.4f}, "
                      f"A1 Consistency={consistency['avg_agent1_consistency']:.2f}, "
                      f"A2 Consistency={consistency['avg_agent2_consistency']:.2f}")
                
                # Print symbol assignment details
                print("\nSymbol assignments:")
                for puzzle_idx, details in consistency['symbol_assignments'].items():
                    print(f"  {puzzle_idx}: "
                          f"A1={details['agent1_symbol']} ({details['agent1_confidence']:.2f}), "
                          f"A2={details['agent2_symbol']} ({details['agent2_confidence']:.2f})")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Entropy={epoch_entropy:.4f}")
        
        return history