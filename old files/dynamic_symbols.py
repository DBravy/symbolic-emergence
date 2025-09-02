import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
import copy

class PlateauDetector:
    """
    Detects when training has plateaued based on accuracy metrics.
    """
    def __init__(
        self,
        plateau_cycles: int = 500,
        plateau_threshold: float = 0.20,  # 20 percentage points
        min_cycles_before_detection: int = 200,
        patience_multiplier: float = 1.0
    ):
        """
        Args:
            plateau_cycles: Number of cycles to check for plateau
            plateau_threshold: Accuracy range that constitutes a plateau (0-1)
            min_cycles_before_detection: Minimum cycles before plateau detection starts
            patience_multiplier: Multiplier for patience when symbols are added
        """
        self.plateau_cycles = plateau_cycles
        self.plateau_threshold = plateau_threshold
        self.min_cycles_before_detection = min_cycles_before_detection
        self.patience_multiplier = patience_multiplier
        
        self.accuracy_history = deque(maxlen=plateau_cycles * 2)  # Keep extra history
        self.cycle_count = 0
        self.last_symbol_addition_cycle = 0
        
    def update(self, accuracy: float) -> bool:
        """
        Update with new accuracy and check for plateau.
        
        Args:
            accuracy: Current accuracy (0-1)
            
        Returns:
            True if plateau detected and new symbol should be added
        """
        self.accuracy_history.append(accuracy)
        self.cycle_count += 1
        
        # Don't detect plateaus too early
        cycles_since_last_addition = self.cycle_count - self.last_symbol_addition_cycle
        if cycles_since_last_addition < self.min_cycles_before_detection:
            return False
            
        # Need enough history for plateau detection
        if len(self.accuracy_history) < self.plateau_cycles:
            return False
            
        # Check if we're in a plateau
        recent_accuracies = list(self.accuracy_history)[-self.plateau_cycles:]
        min_acc = min(recent_accuracies)
        max_acc = max(recent_accuracies)
        accuracy_range = max_acc - min_acc
        
        is_plateau = accuracy_range <= self.plateau_threshold
        
        if is_plateau:
            print(f"\nPlateau detected at cycle {self.cycle_count}:")
            print(f"  Accuracy range over last {self.plateau_cycles} cycles: {accuracy_range:.3f}")
            print(f"  Min accuracy: {min_acc:.3f}, Max accuracy: {max_acc:.3f}")
            print(f"  Threshold: {self.plateau_threshold:.3f}")
            self.last_symbol_addition_cycle = self.cycle_count
            return True
            
        return False
    
    def reset_for_new_symbol(self):
        """Reset detection after adding a new symbol"""
        self.last_symbol_addition_cycle = self.cycle_count
        # Keep history but mark the addition point
        print(f"  Reset plateau detection at cycle {self.cycle_count}")

class PositionalSymbolManager:
    """
    Manages dynamic addition of communication symbols with positional constraints.
    Each position in the sequence gets its own dedicated symbol vocabulary.
    """
    def __init__(
        self,
        initial_total_symbols: int = 12,
        puzzle_symbols: int = 10,
        max_total_symbols: int = 30,
        initial_seq_length: int = 1,
        max_seq_length: int = 10,
        device: str = 'cuda'
    ):
        """
        Args:
            initial_total_symbols: Starting number of total symbols
            puzzle_symbols: Number of puzzle symbols (fixed)
            max_total_symbols: Maximum total symbols allowed
            initial_seq_length: Starting sequence length
            max_seq_length: Maximum sequence length allowed
            device: Device for tensors
        """
        self.initial_total_symbols = initial_total_symbols
        self.puzzle_symbols = puzzle_symbols
        self.max_total_symbols = max_total_symbols
        self.initial_seq_length = initial_seq_length
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.current_total_symbols = initial_total_symbols
        self.current_seq_length = initial_seq_length
        self.symbol_addition_history = []
        
        # NEW: Track symbol generations for positional constraints
        # Each position gets 2 symbols: position 0 gets symbols 0-1, position 1 gets symbols 2-3, etc.
        self.position_to_symbols = {}  # position -> list of symbol indices
        
        # Initialize with first position symbols
        initial_comm_start = self.puzzle_symbols
        self.position_to_symbols[0] = list(range(initial_comm_start, initial_comm_start + 2))
        
    @property
    def current_comm_symbols(self) -> int:
        """Current number of communication symbols"""
        return self.current_total_symbols - self.puzzle_symbols
    
    def get_symbols_for_position(self, position: int) -> List[int]:
        """Get the allowed symbols for a specific sequence position"""
        return self.position_to_symbols.get(position, [])
    
    def get_position_mask(self, position: int, total_comm_symbols: int) -> torch.Tensor:
        """
        Get a mask indicating which communication symbols are allowed at this position.
        
        Args:
            position: Sequence position (0-indexed)
            total_comm_symbols: Total number of communication symbols
            
        Returns:
            Boolean mask of shape [total_comm_symbols] where True = allowed
        """
        mask = torch.zeros(total_comm_symbols, dtype=torch.bool, device=self.device)
        allowed_symbols = self.get_symbols_for_position(position)
        
        if allowed_symbols:
            # Convert absolute symbol indices to communication-only indices
            comm_indices = [s - self.puzzle_symbols for s in allowed_symbols if s >= self.puzzle_symbols]
            for idx in comm_indices:
                if 0 <= idx < total_comm_symbols:
                    mask[idx] = True
        
        return mask
    
    def can_add_symbols(self) -> bool:
        """Check if we can add more symbols and increase sequence length"""
        can_add_symbols = self.current_total_symbols + 2 <= self.max_total_symbols
        can_increase_seq = self.current_seq_length + 1 <= self.max_seq_length
        return can_add_symbols and can_increase_seq
    
    def add_symbols_to_agent(self, agent, cycle: int) -> bool:
        """
        Add 2 new communication symbols for the next sequence position and increase sequence length by 1.
        
        Args:
            agent: Agent to modify
            cycle: Current training cycle
            
        Returns:
            True if symbols were successfully added
        """
        if not self.can_add_symbols():
            print(f"Cannot add symbols: already at maximum ({self.max_total_symbols} symbols, {self.max_seq_length} seq length)")
            return False
            
        old_num_symbols = agent.num_symbols
        new_num_symbols = old_num_symbols + 2  # Add 2 symbols
        old_seq_length = agent.max_seq_length
        new_seq_length = old_seq_length + 1  # Increase sequence length by 1
        
        # NEW: Assign new symbols to the new position
        new_position = self.current_seq_length  # This will be the new position index
        new_symbol_indices = list(range(old_num_symbols, new_num_symbols))
        self.position_to_symbols[new_position] = new_symbol_indices
        
        print(f"\nAdding positional symbols to {agent.agent_id}:")
        print(f"  Old symbols: {old_num_symbols} (comm: {old_num_symbols - self.puzzle_symbols})")
        print(f"  New symbols: {new_num_symbols} (comm: {new_num_symbols - self.puzzle_symbols})")
        print(f"  Old sequence length: {old_seq_length}")
        print(f"  New sequence length: {new_seq_length}")
        print(f"  New position {new_position} gets symbols: {new_symbol_indices}")
        
        # Show current position-to-symbol mapping
        print("  Position-to-symbol mapping:")
        for pos in sorted(self.position_to_symbols.keys()):
            symbols = self.position_to_symbols[pos]
            comm_symbols = [s - self.puzzle_symbols for s in symbols]
            print(f"    Position {pos}: symbols {symbols} (comm: {comm_symbols})")
        
        # Expand the main symbol embedding
        self._expand_embedding(agent.embedding_system.symbol_embedding, new_num_symbols)
        
        # Expand the communication embedding
        old_comm_size = agent.communication_embedding.num_embeddings
        new_comm_size = new_num_symbols - self.puzzle_symbols
        if new_comm_size > old_comm_size:
            self._expand_embedding(agent.communication_embedding, new_comm_size)
        
        # Update agent properties
        agent.num_symbols = new_num_symbols
        agent.max_seq_length = new_seq_length
        agent.communication_vocabulary = set(range(new_num_symbols))
        
        # NEW: Pass position mapping to agent
        agent.position_to_symbols = self.position_to_symbols.copy()
        
        # Update encoder sequence length and rebuild length head if necessary
        if hasattr(agent.encoder, 'max_seq_length'):
            agent.encoder.max_seq_length = new_seq_length
            # NEW: Pass position mapping to encoder
            agent.encoder.position_to_symbols = self.position_to_symbols.copy()
            agent.encoder.puzzle_symbols = self.puzzle_symbols
            # Rebuild the length prediction head with new output size
            self._rebuild_length_head(agent.encoder, new_seq_length)
            
        # Update encoder if it has num_comm_symbols attribute
        if hasattr(agent.encoder, 'num_comm_symbols'):
            agent.encoder.num_comm_symbols = new_comm_size
            
        self.current_total_symbols = new_num_symbols
        self.current_seq_length = new_seq_length
        self.symbol_addition_history.append({
            'cycle': cycle,
            'total_symbols': new_num_symbols,
            'comm_symbols': new_comm_size,
            'seq_length': new_seq_length,
            'position_mapping': self.position_to_symbols.copy()  # Store mapping state
        })
        
        print(f"  Symbol and sequence length addition successful!")
        return True
    
    def _rebuild_length_head(self, encoder, new_seq_length: int):
        """Rebuild the encoder's length prediction head for new sequence length"""
        if hasattr(encoder, 'length_head'):
            old_head = encoder.length_head
            # Create new length head with updated output size
            encoder.length_head = nn.Linear(old_head.in_features, new_seq_length).to(self.device)
            
            # Initialize new head weights
            nn.init.xavier_uniform_(encoder.length_head.weight, gain=1.0)
            if encoder.length_head.bias is not None:
                nn.init.zeros_(encoder.length_head.bias)
            
            print(f"    Rebuilt length head: {old_head.out_features} -> {new_seq_length} outputs")
    
    def _expand_embedding(self, embedding: nn.Embedding, new_size: int):
        """Expand an embedding layer to accommodate more symbols"""
        old_size = embedding.num_embeddings
        if new_size <= old_size:
            return
            
        embedding_dim = embedding.embedding_dim
        
        # Create new embedding layer
        new_embedding = nn.Embedding(new_size, embedding_dim, device=self.device)
        
        # Copy old weights
        with torch.no_grad():
            new_embedding.weight[:old_size] = embedding.weight
            # Initialize new weights with small random values
            nn.init.normal_(new_embedding.weight[old_size:], mean=0, std=0.02)
        
        # Replace the embedding
        embedding.num_embeddings = new_size
        embedding.weight = new_embedding.weight
    
    def get_status(self) -> Dict:
        """Get current status of symbol and sequence length management"""
        return {
            'current_total_symbols': self.current_total_symbols,
            'current_comm_symbols': self.current_comm_symbols,
            'current_seq_length': self.current_seq_length,
            'puzzle_symbols': self.puzzle_symbols,
            'max_total_symbols': self.max_total_symbols,
            'max_seq_length': self.max_seq_length,
            'can_add_more': self.can_add_symbols(),
            'addition_history': self.symbol_addition_history.copy(),
            'position_to_symbols': self.position_to_symbols.copy()
        }

class DynamicTrainingManager:
    """
    Manages the overall dynamic training process with plateau detection and positional symbol/sequence length addition.
    """
    def __init__(
        self,
        agents: List,
        plateau_cycles: int = 500,
        plateau_threshold: float = 0.20,
        min_cycles_before_detection: int = 200,
        initial_total_symbols: int = 12,
        puzzle_symbols: int = 10,
        max_total_symbols: int = 30,
        initial_seq_length: int = 1,
        max_seq_length: int = 10,
        device: str = 'cuda'
    ):
        """
        Args:
            agents: List of agents to manage
            plateau_cycles: Cycles to check for plateau
            plateau_threshold: Accuracy range for plateau (0-1)
            min_cycles_before_detection: Min cycles before plateau detection
            initial_total_symbols: Starting total symbols
            puzzle_symbols: Number of puzzle symbols
            max_total_symbols: Maximum total symbols
            initial_seq_length: Starting sequence length
            max_seq_length: Maximum sequence length
            device: Device for tensors
        """
        self.agents = agents
        self.device = device
        
        self.plateau_detector = PlateauDetector(
            plateau_cycles=plateau_cycles,
            plateau_threshold=plateau_threshold,
            min_cycles_before_detection=min_cycles_before_detection
        )
        
        # Use the new PositionalSymbolManager
        self.symbol_manager = PositionalSymbolManager(
            initial_total_symbols=initial_total_symbols,
            puzzle_symbols=puzzle_symbols,
            max_total_symbols=max_total_symbols,
            initial_seq_length=initial_seq_length,
            max_seq_length=max_seq_length,
            device=device
        )
        
        # Set initial symbol counts and sequence lengths for all agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all agents with the starting symbol and sequence length configuration"""
        for agent in self.agents:
            agent.num_symbols = self.symbol_manager.initial_total_symbols
            agent.max_seq_length = self.symbol_manager.initial_seq_length
            agent.communication_vocabulary = set(range(agent.num_symbols))
            
            # NEW: Initialize position mapping
            agent.position_to_symbols = self.symbol_manager.position_to_symbols.copy()
            
            # Ensure embeddings are correctly sized
            if agent.embedding_system.symbol_embedding.num_embeddings != agent.num_symbols:
                self.symbol_manager._expand_embedding(
                    agent.embedding_system.symbol_embedding, 
                    agent.num_symbols
                )
            
            comm_symbols = agent.num_symbols - self.symbol_manager.puzzle_symbols
            if agent.communication_embedding.num_embeddings != comm_symbols:
                self.symbol_manager._expand_embedding(
                    agent.communication_embedding,
                    comm_symbols
                )
            
            # Initialize encoder sequence length and position mapping
            if hasattr(agent.encoder, 'max_seq_length'):
                agent.encoder.max_seq_length = self.symbol_manager.initial_seq_length
                agent.encoder.position_to_symbols = self.symbol_manager.position_to_symbols.copy()
                agent.encoder.puzzle_symbols = self.symbol_manager.puzzle_symbols
                # Rebuild length head for initial sequence length
                self.symbol_manager._rebuild_length_head(agent.encoder, self.symbol_manager.initial_seq_length)
    
    def update_and_check(self, cycle: int, metrics: Dict) -> bool:
        """
        Update training state and check if symbols should be added.
        
        Args:
            cycle: Current training cycle
            metrics: Dictionary containing training metrics
            
        Returns:
            True if symbols were added
        """
        # Extract average accuracy from metrics
        avg_accuracy = (
            metrics.get('agent1_content_accuracy', 0) + 
            metrics.get('agent2_content_accuracy', 0)
        ) / 2.0
        
        # Check for plateau
        should_add_symbols = self.plateau_detector.update(avg_accuracy)
        
        if should_add_symbols and self.symbol_manager.can_add_symbols():
            # Add 2 symbols and increase sequence length by 1 for all agents
            symbols_added = False
            for agent in self.agents:
                if self.symbol_manager.add_symbols_to_agent(agent, cycle):
                    symbols_added = True
            
            if symbols_added:
                self.plateau_detector.reset_for_new_symbol()
                return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get comprehensive status of dynamic training"""
        status = self.symbol_manager.get_status()
        status.update({
            'plateau_detector': {
                'cycle_count': self.plateau_detector.cycle_count,
                'cycles_since_last_addition': (
                    self.plateau_detector.cycle_count - 
                    self.plateau_detector.last_symbol_addition_cycle
                ),
                'plateau_cycles': self.plateau_detector.plateau_cycles,
                'plateau_threshold': self.plateau_detector.plateau_threshold,
                'min_cycles_before_detection': self.plateau_detector.min_cycles_before_detection,
                'recent_accuracy_range': self._get_recent_accuracy_range()
            }
        })
        return status
    
    def _get_recent_accuracy_range(self) -> Optional[float]:
        """Get the accuracy range over recent cycles"""
        if len(self.plateau_detector.accuracy_history) < 10:
            return None
        
        recent = list(self.plateau_detector.accuracy_history)[-min(100, len(self.plateau_detector.accuracy_history)):]
        return max(recent) - min(recent)