import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Set
from agent_selection import ProgressiveSelectionAgent
from puzzle import Puzzle
import numpy as np
import random
from collections import defaultdict, Counter

class ProgressiveSelectionTrainer:
    def __init__(
        self,
        agent1: ProgressiveSelectionAgent,
        agent2: ProgressiveSelectionAgent,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sync_frequency: int = 50,         # Synchronize every N cycles
        num_distractors: int = 3,         # Number of distractor puzzles
        distractor_strategy: str = 'random',  # 'random', 'similar_size', 'hard'
        first_training_cycles: int = 100,     # Cycles for FIRST training phase
        training_cycles: int = 200,           # Cycles for subsequent training phases
        consolidation_tests: int = 5,         # Number of test cycles in consolidation
        puzzles_per_addition: int = 5,        # Puzzles to add each addition phase
        repetitions_per_puzzle: int = 5,      # How many times to repeat each puzzle
        initial_puzzle_count: int = 5,        # NEW: Initial number of puzzles to start with
        initial_comm_symbols: int = None      # NEW: Initial communication symbols (defaults to initial_puzzle_count)
    ):
        # NEW: Store initial configuration
        self.initial_puzzle_count = initial_puzzle_count
        self.initial_comm_symbols = initial_comm_symbols if initial_comm_symbols is not None else initial_puzzle_count
        
        # Update agents with initial communication symbols before moving to device
        agent1.set_initial_comm_symbols(self.initial_comm_symbols)
        agent2.set_initial_comm_symbols(self.initial_comm_symbols)
        
        self.agent1 = agent1.to(device)
        self.agent2 = agent2.to(device)
        self.device = device
        
        # Phase-based training parameters
        self.first_training_cycles = first_training_cycles
        self.training_cycles = training_cycles
        self.consolidation_tests = consolidation_tests
        self.puzzles_per_addition = puzzles_per_addition
        self.repetitions_per_puzzle = repetitions_per_puzzle
        self.cycle_count = 0
        
        # Current phase tracking
        self.current_phase = "pretraining"
        self.phase_cycle = 0
        self.global_phase_count = 0
        
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
        
        # Puzzle and symbol management
        self.active_puzzles = []
        self.available_arc_puzzles = []
        self.puzzle_symbol_mapping = {}
        self.symbol_puzzle_mapping = {}
        self.next_available_symbol = None
        
        # Consolidation tracking
        self.consolidation_results = []
        self.removed_symbols = set()
    
    def set_puzzle_dataset(self, puzzles: List[Puzzle]):
        """Set the full ARC puzzle dataset"""
        self.available_arc_puzzles = puzzles
        print(f"Loaded {len(puzzles)} total ARC puzzles for iterative training")
    
    def initialize_first_puzzles(self):
        """
        Initialize the first set of active puzzles with symbol assignments.
        Now uses the configured initial_puzzle_count instead of a parameter.
        """
        if len(self.available_arc_puzzles) < self.initial_puzzle_count:
            raise ValueError(f"Need at least {self.initial_puzzle_count} puzzles to start")
        
        # Take first puzzles
        self.active_puzzles = self.available_arc_puzzles[:self.initial_puzzle_count]
        
        # Create symbol assignments starting from puzzle_symbols
        self.puzzle_symbol_mapping = {}
        self.symbol_puzzle_mapping = {}
        
        start_symbol = self.agent1.puzzle_symbols
        for i, puzzle in enumerate(self.active_puzzles):
            symbol_idx = start_symbol + i
            self.puzzle_symbol_mapping[i] = symbol_idx
            self.symbol_puzzle_mapping[symbol_idx] = i
        
        self.next_available_symbol = start_symbol + len(self.active_puzzles)
        
        # Update agent vocabularies to match actual puzzle count
        actual_comm_symbols = len(self.active_puzzles)
        self.agent1.current_comm_symbols = actual_comm_symbols
        self.agent2.current_comm_symbols = actual_comm_symbols
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + actual_comm_symbols
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + actual_comm_symbols
        
        print(f"Initialized with {len(self.active_puzzles)} puzzles")
        print(f"Symbol assignments: {self.puzzle_symbol_mapping}")
        print(f"Next available symbol: {self.next_available_symbol}")
        print(f"Agent communication symbols updated to: {actual_comm_symbols}")
    
    def should_synchronize(self) -> bool:
        """Check if it's time to synchronize agent parameters"""
        return (self.cycle_count > 0 and 
                self.cycle_count - self.last_sync_cycle >= self.sync_frequency)
        
    
    def get_training_cycles_for_current_phase(self) -> int:
        """
        Get the appropriate number of training cycles for the current training phase.
        Returns fewer cycles for the first training phase, more for subsequent ones.
        """
        if self.global_phase_count == 0:
            # First training phase - use shorter cycle count
            return self.first_training_cycles
        else:
            # Subsequent training phases - use standard cycle count
            return self.training_cycles
    
    def synchronize_agents(self):
        """Synchronize ALL agent parameters from agent1 to agent2."""
        print(f"\n{'='*40}")
        print(f"SYNCHRONIZING AGENTS")
        print(f"{'='*40}")
        
        with torch.no_grad():
            total_synced_params = 0
            
            # 1. Synchronize puzzle embedding system
            total_synced_params += self._sync_module_recursive(
                self.agent1.embedding_system, 
                self.agent2.embedding_system
            )
            
            # 2. Synchronize encoder
            total_synced_params += self._sync_module_recursive(
                self.agent1.encoder, 
                self.agent2.encoder
            )
            
            # 3. Synchronize message pooling
            total_synced_params += self._sync_module_recursive(
                self.agent1.message_pooling,
                self.agent2.message_pooling
            )
            
            # 4. Synchronize similarity MLP if it exists
            if hasattr(self.agent1, 'similarity_mlp') and hasattr(self.agent2, 'similarity_mlp'):
                total_synced_params += self._sync_module_recursive(
                    self.agent1.similarity_mlp,
                    self.agent2.similarity_mlp
                )
            
            # 5. Synchronize communication embeddings (active symbols only)
            current_comm_symbols = self.agent1.current_comm_symbols
            if current_comm_symbols > 0:
                start_idx = self.agent1.puzzle_symbols
                end_idx = start_idx + current_comm_symbols
                
                self.agent2.communication_embedding.weight[start_idx:end_idx].copy_(
                    self.agent1.communication_embedding.weight[start_idx:end_idx]
                )
                comm_params = current_comm_symbols * self.agent1.communication_embedding.embedding_dim
                total_synced_params += comm_params
            
            # Update sync tracking
            self.last_sync_cycle = self.cycle_count
            
            print(f"Synchronized {total_synced_params:,} parameters")
    
    def _sync_module_recursive(self, module1, module2):
        """Recursively synchronize all parameters in a module"""
        synced_count = 0
        
        # Sync direct parameters
        for name, param1 in module1.named_parameters(recurse=False):
            if hasattr(module2, name.split('.')[-1]):
                param2 = getattr(module2, name.split('.')[-1])
                if hasattr(param2, 'copy_'):
                    param2.copy_(param1)
                    synced_count += param1.numel()
        
        # Sync buffers
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
                synced_count += self._sync_module_recursive(child1, child2)
        
        return synced_count
    
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
    
    def _set_component_trainable(self, agent, component_name, trainable):
        """Helper method to set requires_grad for a component's parameters"""
        if hasattr(agent, component_name):
            component = getattr(agent, component_name)
            for param in component.parameters():
                param.requires_grad = trainable

    def sample_distractors(self, target_puzzle: torch.Tensor, target_idx: int) -> List[torch.Tensor]:
        """Modified to sample from ALL active puzzles, not just mapped ones"""
        if len(self.active_puzzles) < self.num_distractors + 1:
            raise ValueError(f"Need at least {self.num_distractors + 1} active puzzles for selection task")
        
        distractors = []
        available_indices = list(range(len(self.active_puzzles)))
        available_indices.remove(target_idx)
        
        if self.distractor_strategy == 'random':
            distractor_indices = random.sample(available_indices, self.num_distractors)
        elif self.distractor_strategy == 'similar_size':
            # Similar size sampling
            target_height, target_width = target_puzzle.shape[1], target_puzzle.shape[2]
            size_diffs = []
            for idx in available_indices:
                puzzle = self.active_puzzles[idx]
                puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long)
                h_diff = abs(puzzle_tensor.shape[0] - target_height)
                w_diff = abs(puzzle_tensor.shape[1] - target_width)
                size_diff = h_diff + w_diff
                size_diffs.append((size_diff, idx))
            
            size_diffs.sort(key=lambda x: x[0])
            distractor_indices = [idx for _, idx in size_diffs[:self.num_distractors]]
        else:
            distractor_indices = random.sample(available_indices, self.num_distractors)
        
        # Convert to tensors - include ALL puzzles as potential distractors
        for idx in distractor_indices:
            puzzle = self.active_puzzles[idx]
            distractor_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0)
            distractors.append(distractor_tensor)
        
        return distractors

    def run_consolidation_test(self) -> Dict[str, List]:
        """
        Run consolidation tests to identify recessive symbols.
        Returns confusion matrix data for analysis.
        """
        print(f"\n{'='*50}")
        print(f"CONSOLIDATION PHASE - Testing Symbol Accuracy")
        print(f"{'='*50}")
        
        # Set agents to eval mode
        self.agent1.eval()
        self.agent2.eval()
        
        # Track results for each test
        confusion_data = defaultdict(list)  # symbol -> [predicted_symbols_list]
        
        with torch.no_grad():
            for test_round in range(self.consolidation_tests):
                print(f"\nConsolidation Test Round {test_round + 1}/{self.consolidation_tests}")
                
                round_results = {}
                
                mapped_puzzle_indices = list(self.puzzle_symbol_mapping.keys())
                print(f"Testing {len(mapped_puzzle_indices)} puzzles with symbol mappings")
                print(f"Skipping {len(self.active_puzzles) - len(mapped_puzzle_indices)} puzzles without mappings")

                if len(mapped_puzzle_indices) == 0:
                    print("No puzzles with symbol mappings to test!")
                    return {}

                for puzzle_idx in mapped_puzzle_indices:
                    puzzle = self.active_puzzles[puzzle_idx]
                    puzzle_tensor = torch.tensor(
                        puzzle.test_input, 
                        dtype=torch.long, 
                        device=self.device
                    ).unsqueeze(0)
                    
                    assigned_symbol = self.puzzle_symbol_mapping[puzzle_idx]
                    
                    # Agent1 encodes the puzzle
                    symbols, _, _ = self.agent1.encode_puzzle_to_message(
                        puzzle_tensor, temperature=0.1, deterministic=True
                    )
                    
                    # Create candidates with all active puzzles
                    candidates = []
                    for other_puzzle in self.active_puzzles:
                        candidate_tensor = torch.tensor(
                            other_puzzle.test_input,
                            dtype=torch.long,
                            device=self.device
                        ).unsqueeze(0)
                        candidates.append(candidate_tensor)
                    
                    # Agent2 selects from all puzzles
                    selection_probs, selection_logits, _ = self.agent2.select_from_candidates(
                        symbols, candidates, temperature=0.1
                    )
                    
                    predicted_idx = selection_logits.argmax(dim=-1).item()
                    
                    # FIX: Check if predicted puzzle has a symbol mapping
                    if predicted_idx in self.puzzle_symbol_mapping:
                        predicted_symbol = self.puzzle_symbol_mapping[predicted_idx]
                    else:
                        # Handle case where predicted puzzle doesn't have a symbol mapping
                        predicted_symbol = -1  # Use -1 to indicate "no symbol"
                    
                    # Record the result
                    confusion_data[assigned_symbol].append(predicted_symbol)
                    round_results[assigned_symbol] = predicted_symbol
                    
                    correct = "✓" if predicted_idx == puzzle_idx else "✗"
                    if predicted_symbol != -1:
                        symbol_info = f"Symbol {predicted_symbol}"
                    else:
                        symbol_info = "No symbol mapping"
                    
                    print(f"  Puzzle {puzzle_idx} (Symbol {assigned_symbol}): "
                        f"Predicted {predicted_idx} ({symbol_info}) {correct}")
        
        return dict(confusion_data)
    
    def identify_recessive_symbols(self, confusion_data: Dict[int, List[int]]) -> Set[int]:
        """
        Identify symbols that are consistently misinterpreted.
        A symbol is considered recessive if it's never predicted correctly
        across all test rounds.
        """
        recessive_symbols = set()
        
        print(f"\n{'='*50}")
        print(f"ANALYZING SYMBOL PERFORMANCE")
        print(f"{'='*50}")
        
        for symbol, predictions in confusion_data.items():
            correct_predictions = sum(1 for pred in predictions if pred == symbol)
            accuracy = correct_predictions / len(predictions)
            
            print(f"Symbol {symbol}: {correct_predictions}/{len(predictions)} correct ({accuracy:.2f})")
            
            # Consider symbol recessive if accuracy is below threshold
            if accuracy == 0.0:  # Never predicted correctly
                recessive_symbols.add(symbol)
                print(f"  → RECESSIVE: Symbol {symbol} never predicted correctly")
            
            # Show what this symbol was confused with
            if accuracy < 1.0:
                confusion_counter = Counter(predictions)
                most_common = confusion_counter.most_common(2)
                print(f"  → Most often confused with: {most_common}")
        
        return recessive_symbols
    
    def remove_recessive_symbols(self, recessive_symbols: Set[int]):
        """Remove recessive symbols but keep their associated puzzles"""
        if not recessive_symbols:
            print("No recessive symbols to remove")
            return
        
        print(f"\n{'='*50}")
        print(f"REMOVING RECESSIVE SYMBOLS: {recessive_symbols}")
        print(f"{'='*50}")
        
        # Find puzzles that will lose their symbol mappings
        orphaned_puzzles = []
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                orphaned_puzzles.append(puzzle_idx)
        
        print(f"Puzzles that will lose symbol mappings: {orphaned_puzzles}")
        print(f"These puzzles will remain active but won't have dedicated symbols")
        
        # STORE ORIGINAL MAPPINGS BEFORE REMOVAL
        original_puzzle_symbol_mapping = self.puzzle_symbol_mapping.copy()
        
        # Remove symbol mappings for recessive symbols
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                print(f"Removing symbol mapping: Puzzle {puzzle_idx} <-> Symbol {symbol}")
                
                # Remove from both mappings
                del self.symbol_puzzle_mapping[symbol]
                del self.puzzle_symbol_mapping[puzzle_idx]
        
        # Compact remaining symbol mappings to be sequential
        # This ensures communication symbols are contiguous
        remaining_mapped_puzzles = list(self.puzzle_symbol_mapping.keys())
        remaining_mapped_puzzles.sort()
        
        new_puzzle_mapping = {}
        new_symbol_mapping = {}
        new_symbol_idx = self.agent1.puzzle_symbols
        
        # Track symbol transfers for embedding copying
        symbol_transfer_mapping = {}  # old_symbol -> new_symbol
        
        for puzzle_idx in remaining_mapped_puzzles:
            old_symbol = original_puzzle_symbol_mapping[puzzle_idx]
            new_symbol = new_symbol_idx
            
            new_puzzle_mapping[puzzle_idx] = new_symbol
            new_symbol_mapping[new_symbol] = puzzle_idx
            
            # Track the transfer (only if symbol actually changes)
            if old_symbol != new_symbol:
                symbol_transfer_mapping[old_symbol] = new_symbol
                print(f"Symbol transfer: {old_symbol} -> {new_symbol} (puzzle {puzzle_idx})")
            
            new_symbol_idx += 1
        
        # UPDATE MAPPINGS
        self.puzzle_symbol_mapping = new_puzzle_mapping
        self.symbol_puzzle_mapping = new_symbol_mapping
        
        # TRANSFER EMBEDDINGS FOR REMAPPED SYMBOLS
        if symbol_transfer_mapping:
            print(f"\nTransferring embeddings for {len(symbol_transfer_mapping)} remapped symbols...")
            
            with torch.no_grad():
                # Transfer embeddings for Agent 1
                for old_symbol, new_symbol in symbol_transfer_mapping.items():
                    self.agent1.communication_embedding.weight[new_symbol].copy_(
                        self.agent1.communication_embedding.weight[old_symbol]
                    )
                    print(f"  Agent1: Copied embedding {old_symbol} -> {new_symbol}")
                
                # Transfer embeddings for Agent 2
                for old_symbol, new_symbol in symbol_transfer_mapping.items():
                    self.agent2.communication_embedding.weight[new_symbol].copy_(
                        self.agent2.communication_embedding.weight[old_symbol]
                    )
                    print(f"  Agent2: Copied embedding {old_symbol} -> {new_symbol}")
            
            print(f"Embedding transfer complete: {len(symbol_transfer_mapping)} transfers")
        else:
            print("No embedding transfers needed (symbols remain in same positions)")
        
        # Update agent vocabularies based on remaining mapped symbols
        self.agent1.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent2.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + len(self.puzzle_symbol_mapping)
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + len(self.puzzle_symbol_mapping)
        
        # Track removed symbols
        self.removed_symbols.update(recessive_symbols)
        
        print(f"\nResults:")
        print(f"  Total puzzles (unchanged): {len(self.active_puzzles)}")
        print(f"  Puzzles with symbol mappings: {len(self.puzzle_symbol_mapping)}")
        print(f"  Puzzles without symbol mappings: {len(self.active_puzzles) - len(self.puzzle_symbol_mapping)}")
        print(f"  Active communication symbols: {self.agent1.current_comm_symbols}")
        print(f"  New symbol mapping: {self.puzzle_symbol_mapping}")
        print(f"  Embeddings preserved for surviving symbols: ✓")
    
    def add_new_puzzles(self):
        """Add new puzzles from the ARC dataset"""
        print(f"\n{'='*50}")
        print(f"ADDITION PHASE - Adding {self.puzzles_per_addition} new puzzles")
        print(f"{'='*50}")
        
        # Find puzzles not yet used
        used_puzzle_count = len(self.active_puzzles) + len(self.removed_symbols)
        if used_puzzle_count + self.puzzles_per_addition > len(self.available_arc_puzzles):
            actual_addition = len(self.available_arc_puzzles) - used_puzzle_count
            print(f"Only {actual_addition} puzzles available to add")
            self.puzzles_per_addition = actual_addition
        
        if self.puzzles_per_addition <= 0:
            print("No more puzzles to add!")
            return []
        
        # Add new puzzles
        start_idx = len(self.active_puzzles)
        new_puzzles = []
        
        for i in range(self.puzzles_per_addition):
            puzzle_idx = used_puzzle_count + i
            if puzzle_idx < len(self.available_arc_puzzles):
                new_puzzle = self.available_arc_puzzles[puzzle_idx]
                self.active_puzzles.append(new_puzzle)
                new_puzzles.append(new_puzzle)
                
                # Assign symbol
                active_puzzle_idx = len(self.active_puzzles) - 1
                symbol_idx = self.agent1.puzzle_symbols + active_puzzle_idx
                self.puzzle_symbol_mapping[active_puzzle_idx] = symbol_idx
                self.symbol_puzzle_mapping[symbol_idx] = active_puzzle_idx
                
                print(f"Added puzzle {puzzle_idx} as active puzzle {active_puzzle_idx} with symbol {symbol_idx}")
        
        # Update agent vocabularies
        self.agent1.current_comm_symbols = len(self.active_puzzles)
        self.agent2.current_comm_symbols = len(self.active_puzzles)
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + len(self.active_puzzles)
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + len(self.active_puzzles)
        
        print(f"Total active puzzles: {len(self.active_puzzles)}")
        print(f"New symbol mapping: {self.puzzle_symbol_mapping}")
        
        return new_puzzles
    
    def advance_phase(self):
        """Advance to the next phase in the cycle"""
        if self.current_phase == "pretraining":
            self.current_phase = "training"
        elif self.current_phase == "training":
            # self.current_phase = "addition"
            self.current_phase = "consolidation"
        elif self.current_phase == "consolidation":
            self.current_phase = "addition"
        elif self.current_phase == "addition":
            self.current_phase = "pretraining"
            self.global_phase_count += 1
        
        self.phase_cycle = 0
        print(f"\n{'='*60}")
        print(f"ADVANCING TO {self.current_phase.upper()} PHASE")
        if self.current_phase == "pretraining" and self.global_phase_count > 0:
            print(f"Global Phase Cycle: {self.global_phase_count}")
        print(f"{'='*60}")

    def train_bidirectional_step(
        self,
        puzzle: torch.Tensor,
        puzzle_idx: int,
        num_exchanges: int = 1,
        temperature: float = 1.0,
        initial_phase: bool = False
    ) -> List[Dict[str, float]]:
        
        # Check if this puzzle has a symbol mapping
        has_symbol_mapping = puzzle_idx in self.puzzle_symbol_mapping
        
        # Check if it's time to synchronize agents
        if self.should_synchronize():
            self.synchronize_agents()
        
        # Increment cycle count
        self.cycle_count += 1
        self.phase_cycle += 1
        
        metrics_history = []
        
        # Set training mode based on phase
        if self.current_phase == "pretraining":
            self.set_training_mode("encoder_only")
        else:
            self.set_training_mode("joint")
        
        for exchange in range(num_exchanges):
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            
            # Sample distractors for this exchange (from ALL active puzzles)
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
            # But only if this puzzle has a symbol mapping
            if self.training_mode in ["encoder_only", "joint"] and has_symbol_mapping:
                entropy1 = self._calculate_symbol_entropy(symbols1)
                entropy2 = self._calculate_symbol_entropy(symbols2)
                
                entropy_weight = 0.1
                total_loss = total_loss - entropy_weight * (entropy1 + entropy2)
            
            total_loss.backward()
            
            self.opt1.step()
            self.opt2.step()
            
            # Compute accuracies
            with torch.no_grad():
                # Selection accuracy
                pred1 = selection_logits1.argmax(dim=-1)
                pred2 = selection_logits2.argmax(dim=-1)
                
                acc1 = (pred1 == target_idx[0]).float().mean().item()
                acc2 = (pred2 == target_idx[0]).float().mean().item()
                
                # Confidence in correct selection
                correct_confidence1 = selection_probs1[0, 0].item()
                correct_confidence2 = selection_probs2[0, 0].item()
            
            metrics = {
                'cycle': self.cycle_count,
                'phase': self.current_phase,
                'phase_cycle': self.phase_cycle,
                'global_phase_count': self.global_phase_count,
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
                'active_puzzles': len(self.active_puzzles),
                'mapped_puzzles': len(self.puzzle_symbol_mapping),
                'unmapped_puzzles': len(self.active_puzzles) - len(self.puzzle_symbol_mapping),
                'target_has_mapping': has_symbol_mapping
            }
            
            metrics_history.append(metrics)
            
        return metrics_history
    
    def _calculate_symbol_entropy(self, symbol_probs):
        """Calculate the entropy of symbol probability distributions."""
        eps = 1e-8
        probs = symbol_probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean()

    def get_phase_status(self) -> Dict[str, any]:
            """Get current phase and vocabulary status"""
            return {
                'current_phase': self.current_phase,
                'phase_cycle': self.phase_cycle,
                'global_phase_count': self.global_phase_count,
                'active_puzzles': len(self.active_puzzles),
                'removed_symbols': len(self.removed_symbols),
                'agent1_vocab': self.agent1.get_vocabulary_info(),
                'agent2_vocab': self.agent2.get_vocabulary_info(),
                'puzzle_symbol_mapping': self.puzzle_symbol_mapping,
                'selection_config': {
                    'num_distractors': self.num_distractors,
                    'distractor_strategy': self.distractor_strategy,
                    'first_training_cycles': self.first_training_cycles,
                    'training_cycles': self.training_cycles,
                    'consolidation_tests': self.consolidation_tests,
                    'puzzles_per_addition': self.puzzles_per_addition,
                    'repetitions_per_puzzle': self.repetitions_per_puzzle,
                    'initial_puzzle_count': self.initial_puzzle_count,        # NEW
                    'initial_comm_symbols': self.initial_comm_symbols        # NEW
                }
            }


# Create a factory function to replace the original CommunicationTrainer
def CommunicationTrainer(*args, **kwargs):
    """Factory function to create ProgressiveSelectionTrainer instances"""
    return ProgressiveSelectionTrainer(*args, **kwargs)