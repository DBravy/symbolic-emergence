import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from retrieval_agent import RetrievalProgressiveAgent
from trainer import ProgressiveCommunicationTrainer
from puzzle import Puzzle
import numpy as np

class RetrievalProgressiveCommunicationTrainer(ProgressiveCommunicationTrainer):
    """
    Extended trainer that supports two-phase training:
    Phase 1: Retrieval mode - agents must select from existing puzzles
    Phase 2: Generation mode - agents generate puzzles from scratch
    """
    def __init__(
        self,
        agent1: RetrievalProgressiveAgent,
        agent2: RetrievalProgressiveAgent,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        expansion_frequency: int = 200,  # Expand every N cycles
        symbols_per_expansion: int = 2,   # Add N symbols per expansion
        length_per_expansion: int = 1,    # Add N sequence length per expansion
        puzzle_database: List[torch.Tensor] = None
    ):
        # Initialize parent class
        super().__init__(
            agent1=agent1,
            agent2=agent2,
            learning_rate=learning_rate,
            device=device,
            expansion_frequency=expansion_frequency,
            symbols_per_expansion=symbols_per_expansion,
            length_per_expansion=length_per_expansion
        )
        
        # Retrieval-specific initialization
        self.puzzle_database = puzzle_database if puzzle_database is not None else []
        self.puzzle_embeddings = None
        self.current_phase = "retrieval"  # Start in retrieval phase
        
        # Phase tracking
        self.phase_history = []
        self.retrieval_cycles = 0
        self.generation_cycles = 0
        
        if self.puzzle_database:
            self._compute_puzzle_embeddings()
            self._setup_puzzle_database()
        
        print(f"Initialized RetrievalProgressiveCommunicationTrainer")
        print(f"  Database size: {len(self.puzzle_database)} puzzles")
        print(f"  Starting phase: {self.current_phase}")
    
    def set_puzzle_database(self, puzzle_database: List[torch.Tensor]):
        """Set or update the puzzle database"""
        self.puzzle_database = puzzle_database
        self._compute_puzzle_embeddings()
        self._setup_puzzle_database()
        print(f"Updated puzzle database with {len(puzzle_database)} puzzles")
    
    def _compute_puzzle_embeddings(self):
        """Compute embeddings for all puzzles in the database"""
        if not self.puzzle_database:
            return
        
        print("Computing puzzle embeddings...")
        embeddings = []
        
        with torch.no_grad():
            for i, puzzle in enumerate(self.puzzle_database):
                # Add batch dimension if needed
                if len(puzzle.shape) == 2:
                    puzzle_batch = puzzle.unsqueeze(0)
                else:
                    puzzle_batch = puzzle
                
                # Move to device
                puzzle_batch = puzzle_batch.to(self.device)
                
                # Get embedding using agent1's embedding system
                embedding = self.agent1.embedding_system.embed_puzzle(puzzle_batch)
                
                # Debug: print shapes for first few puzzles
                if i < 3:
                    print(f"  Puzzle {i}: shape {puzzle.shape} -> embedding shape {embedding.shape}")
                
                # Pool embedding to fixed size - global average pooling over spatial dimensions
                # embedding shape is typically [batch_size, height*width, embedding_dim]
                # We want [batch_size, embedding_dim]
                if len(embedding.shape) == 3:
                    pooled_embedding = embedding.mean(dim=1)  # Average over spatial dimension
                elif len(embedding.shape) == 2:
                    pooled_embedding = embedding  # Already correct shape
                else:
                    raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
                
                # Remove batch dimension and store
                embeddings.append(pooled_embedding.squeeze(0))  # [embedding_dim]
                
                if (i + 1) % 10 == 0:
                    print(f"  Computed {i + 1}/{len(self.puzzle_database)} embeddings")
        
        self.puzzle_embeddings = torch.stack(embeddings, dim=0)  # [num_puzzles, embedding_dim]
        print(f"Computed embeddings shape: {self.puzzle_embeddings.shape}")
    
    def _setup_puzzle_database(self):
        """Set up puzzle database in both agents"""
        if self.puzzle_embeddings is None:
            print("Warning: No puzzle embeddings available")
            return
        
        # Convert puzzles to device if needed
        device_puzzles = []
        for puzzle in self.puzzle_database:
            if puzzle.device != self.device:
                device_puzzles.append(puzzle.to(self.device))
            else:
                device_puzzles.append(puzzle)
        
        # Set database in both agents
        self.agent1.set_puzzle_database(device_puzzles, self.puzzle_embeddings)
        self.agent2.set_puzzle_database(device_puzzles, self.puzzle_embeddings)
        
        print(f"Set up puzzle database in both agents")
    
    def set_training_phase(self, phase: str):
        """Set training phase: 'retrieval' or 'generation'"""
        valid_phases = ["retrieval", "generation"]
        if phase not in valid_phases:
            raise ValueError(f"Phase must be one of {valid_phases}")
        
        old_phase = self.current_phase
        self.current_phase = phase
        
        # Set retrieval mode in both agents
        retrieval_enabled = (phase == "retrieval")
        self.agent1.set_retrieval_mode(retrieval_enabled)
        self.agent2.set_retrieval_mode(retrieval_enabled)
        
        # Record phase change
        self.phase_history.append({
            'cycle': self.cycle_count,
            'old_phase': old_phase,
            'new_phase': phase
        })
        
        print(f"\n{'='*60}")
        print(f"TRAINING PHASE CHANGE AT CYCLE {self.cycle_count}")
        print(f"  {old_phase.upper()} → {phase.upper()}")
        print(f"  Retrieval mode: {'ENABLED' if retrieval_enabled else 'DISABLED'}")
        print(f"{'='*60}")
    
    def train_retrieval_phase(self, cycles: int, **kwargs) -> List[Dict[str, float]]:
        """Train in retrieval phase (constrained to existing puzzles)"""
        print(f"\n--- Starting Retrieval Phase Training ({cycles} cycles) ---")
        self.set_training_phase("retrieval")
        
        metrics_history = []
        for cycle in range(cycles):
            cycle_metrics = self._train_single_cycle(**kwargs)
            metrics_history.extend(cycle_metrics)
            self.retrieval_cycles += 1
            
            # Add phase information to metrics
            for metrics in cycle_metrics:
                metrics['training_phase'] = 'retrieval'
                metrics['retrieval_cycles'] = self.retrieval_cycles
                metrics['generation_cycles'] = self.generation_cycles
        
        print(f"Completed retrieval phase training: {self.retrieval_cycles} total cycles")
        return metrics_history
    
    def train_generation_phase(self, cycles: int, **kwargs) -> List[Dict[str, float]]:
        """Train in generation phase (free puzzle generation)"""
        print(f"\n--- Starting Generation Phase Training ({cycles} cycles) ---")
        self.set_training_phase("generation")
        
        metrics_history = []
        for cycle in range(cycles):
            cycle_metrics = self._train_single_cycle(**kwargs)
            metrics_history.extend(cycle_metrics)
            self.generation_cycles += 1
            
            # Add phase information to metrics
            for metrics in cycle_metrics:
                metrics['training_phase'] = 'generation'
                metrics['retrieval_cycles'] = self.retrieval_cycles
                metrics['generation_cycles'] = self.generation_cycles
        
        print(f"Completed generation phase training: {self.generation_cycles} total cycles")
        return metrics_history
    
    def _train_single_cycle(self, test_puzzles=None, **kwargs):
        """Train for one cycle - wrapper around parent class method"""
        if test_puzzles is None:
            # Use puzzle database as test puzzles in retrieval mode
            if self.current_phase == "retrieval" and self.puzzle_database:
                test_puzzles = [self._tensor_to_puzzle(p) for p in self.puzzle_database]
            else:
                raise ValueError("No test puzzles provided for training")
        
        # Call parent class training method
        return self._train_cycle_with_puzzles(test_puzzles, **kwargs)
    
    def _tensor_to_puzzle(self, tensor: torch.Tensor) -> Puzzle:
        """Convert a tensor to a Puzzle object"""
        # Convert to numpy and create puzzle
        if len(tensor.shape) == 3:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present
        
        puzzle_array = tensor.cpu().numpy()
        return Puzzle.from_single_example(puzzle_array, puzzle_array)
    
    def _train_cycle_with_puzzles(self, test_puzzles, **kwargs):
        """Modified training cycle that works with puzzle list"""
        cycle_metrics = []
        
        for puzzle_idx, puzzle in enumerate(test_puzzles):
            puzzle_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0)
            
            step_metrics = self.train_bidirectional_step(
                puzzle_tensor, 
                num_exchanges=1,
                temperature=1.0,
                initial_phase=False
            )
            
            cycle_metrics.extend(step_metrics)
        
        return cycle_metrics
    
    def train_two_phase(
        self, 
        retrieval_cycles: int = 1000, 
        generation_cycles: int = 1000,
        test_puzzles: List = None,
        **kwargs
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """
        Train using two-phase approach:
        1. Retrieval phase: Learn to select existing puzzles
        2. Generation phase: Learn to generate new puzzles
        """
        print(f"\n{'='*80}")
        print(f"STARTING TWO-PHASE TRAINING")
        print(f"  Phase 1 (Retrieval): {retrieval_cycles} cycles")
        print(f"  Phase 2 (Generation): {generation_cycles} cycles")
        print(f"  Database size: {len(self.puzzle_database)} puzzles")
        print(f"{'='*80}")
        
        # Phase 1: Retrieval training
        retrieval_metrics = self.train_retrieval_phase(
            cycles=retrieval_cycles,
            test_puzzles=test_puzzles,
            **kwargs
        )
        
        # Phase 2: Generation training
        generation_metrics = self.train_generation_phase(
            cycles=generation_cycles,
            test_puzzles=test_puzzles,
            **kwargs
        )
        
        # Print final summary
        print(f"\n{'='*80}")
        print(f"TWO-PHASE TRAINING COMPLETE")
        print(f"  Total retrieval cycles: {self.retrieval_cycles}")
        print(f"  Total generation cycles: {self.generation_cycles}")
        print(f"  Total cycles: {self.retrieval_cycles + self.generation_cycles}")
        print(f"{'='*80}")
        
        return retrieval_metrics, generation_metrics
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        agent1_stats = self.agent1.get_retrieval_stats()
        agent2_stats = self.agent2.get_retrieval_stats()
        
        return {
            'current_phase': self.current_phase,
            'cycle_count': self.cycle_count,
            'retrieval_cycles': self.retrieval_cycles,
            'generation_cycles': self.generation_cycles,
            'database_size': len(self.puzzle_database),
            'agent1_retrieval': agent1_stats,
            'agent2_retrieval': agent2_stats,
            'phase_history': self.phase_history,
            'vocab_status': self.get_vocabulary_status()
        }
    
    def evaluate_retrieval_accuracy(self, test_puzzles: List[torch.Tensor], num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate how accurately agents can retrieve correct puzzles
        
        Args:
            test_puzzles: List of puzzles to test retrieval on
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with accuracy statistics
        """
        if self.current_phase != "retrieval":
            print("Warning: Evaluating retrieval accuracy outside of retrieval phase")
        
        print(f"Evaluating retrieval accuracy on {num_samples} samples...")
        
        correct_retrievals_1 = 0
        correct_retrievals_2 = 0
        total_samples = min(num_samples, len(test_puzzles))
        
        with torch.no_grad():
            for i in range(total_samples):
                puzzle = test_puzzles[i % len(test_puzzles)]
                if len(puzzle.shape) == 2:
                    puzzle = puzzle.unsqueeze(0)
                puzzle = puzzle.to(self.device)
                
                # Agent 1: encode then decode
                message1, _, _ = self.agent1.encode_puzzle_to_message(puzzle, temperature=0.1)
                reconstructed1, _, _, _, _ = self.agent2.decode_message_to_puzzle(message1, temperature=0.1)
                
                # Agent 2: encode then decode
                message2, _, _ = self.agent2.encode_puzzle_to_message(puzzle, temperature=0.1)
                reconstructed2, _, _, _, _ = self.agent1.decode_message_to_puzzle(message2, temperature=0.1)
                
                # Check if reconstruction matches original
                original_discrete = puzzle.squeeze(0)
                recon1_discrete = reconstructed1.argmax(dim=-1).squeeze(0)
                recon2_discrete = reconstructed2.argmax(dim=-1).squeeze(0)
                
                # Compare original size regions
                min_h1 = min(original_discrete.shape[0], recon1_discrete.shape[0])
                min_w1 = min(original_discrete.shape[1], recon1_discrete.shape[1])
                min_h2 = min(original_discrete.shape[0], recon2_discrete.shape[0])
                min_w2 = min(original_discrete.shape[1], recon2_discrete.shape[1])
                
                if torch.equal(original_discrete[:min_h1, :min_w1], recon1_discrete[:min_h1, :min_w1]):
                    correct_retrievals_1 += 1
                
                if torch.equal(original_discrete[:min_h2, :min_w2], recon2_discrete[:min_h2, :min_w2]):
                    correct_retrievals_2 += 1
        
        accuracy_1 = correct_retrievals_1 / total_samples
        accuracy_2 = correct_retrievals_2 / total_samples
        
        results = {
            'agent1_retrieval_accuracy': accuracy_1,
            'agent2_retrieval_accuracy': accuracy_2,
            'average_retrieval_accuracy': (accuracy_1 + accuracy_2) / 2,
            'total_samples': total_samples,
            'correct_retrievals_1': correct_retrievals_1,
            'correct_retrievals_2': correct_retrievals_2
        }
        
        print(f"Retrieval Accuracy Results:")
        print(f"  Agent 1: {accuracy_1:.3f} ({correct_retrievals_1}/{total_samples})")
        print(f"  Agent 2: {accuracy_2:.3f} ({correct_retrievals_2}/{total_samples})")
        print(f"  Average: {results['average_retrieval_accuracy']:.3f}")
        
        return results

# Factory function to create RetrievalProgressiveCommunicationTrainer instances
def RetrievalCommunicationTrainer(*args, **kwargs):
    """Factory function to create RetrievalProgressiveCommunicationTrainer instances"""
    return RetrievalProgressiveCommunicationTrainer(*args, **kwargs)