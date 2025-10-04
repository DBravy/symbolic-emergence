import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Set, Optional
import os
import json
from datetime import datetime
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
        consolidation_tests: int = 10,         # Number of test cycles in consolidation
        consolidation_threshold: float = 0.3,  # Threshold for identifying recessive symbols (30% by default)
        puzzles_per_addition: int = 5,        # Puzzles to add each addition phase
        repetitions_per_puzzle: int = 5,      # How many times to repeat each puzzle
        initial_puzzle_count: int = 5,        # NEW: Initial number of puzzles to start with
        initial_comm_symbols: int = None,     # NEW: Initial communication symbols (defaults to initial_puzzle_count)
        # NEW: Phase-change indicator toggle and novel test config
        phase_change_indicator: str = 'ges',  # 'ges' or 'novel_test'
        novel_test_interval_cycles: int = 10,
        novel_test_threshold_correct: int = 110,
        novel_test_bidirectional: bool = True,
        novel_test_log_summary_only: bool = True,
        web_mode: bool = False                # NEW: Reduced logging for web interface
    ):
        # NEW: Store initial configuration
        self.initial_puzzle_count = initial_puzzle_count
        self.initial_comm_symbols = initial_comm_symbols if initial_comm_symbols is not None else initial_puzzle_count
        self.web_mode = web_mode
        
        # Update agents with initial communication symbols before moving to device
        agent1.set_initial_comm_symbols(self.initial_comm_symbols)
        agent2.set_initial_comm_symbols(self.initial_comm_symbols)
        
        self.agent1 = agent1.to(device)
        self.agent2 = agent2.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Phase-based training parameters
        self.first_training_cycles = first_training_cycles
        self.training_cycles = training_cycles
        self.consolidation_tests = consolidation_tests
        self.consolidation_threshold = consolidation_threshold
        self.puzzles_per_addition = puzzles_per_addition
        self.repetitions_per_puzzle = repetitions_per_puzzle
        self.cycle_count = 0
        
        # Current phase tracking
        self.current_phase = "pretraining"
        self.phase_cycle = 0
        self.global_phase_count = 0
        
        # Position freezing for progressive sequence training
        self.frozen_positions = []  # Track which positions are frozen
        self.frozen_comm_symbols = 0  # Track how many communication symbols are frozen
        
        # Synchronization parameters
        self.sync_frequency = sync_frequency
        self.last_sync_cycle = 0

        # Selection task parameters
        self.num_distractors = num_distractors
        self.initial_num_distractors = num_distractors
        self.distractor_strategy = distractor_strategy
        # NEW: track distractors used in the preceding training phase
        self.last_training_phase_distractors = num_distractors

        self.used_puzzle_indices = set()

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
        # Runtime reconstruction mode (switchable via control API)
        self.reconstruction_mode = False
        self._opt_decoders = None  # lazily initialized when reconstruction mode is enabled
        
        # Puzzle and symbol management
        self.active_puzzles = []
        self.available_arc_puzzles = []
        self.puzzle_symbol_mapping = {}
        self.symbol_puzzle_mapping = {}
        self.next_available_symbol = None
        
        # Consolidation tracking
        self.consolidation_results = []
        self.removed_symbols = set()
        
        # NEW: Moving averages for GES early-stop trigger
        self._ges_window = 50
        self._ges1_values = []
        self._ges2_values = []
        
        # NEW: Early-stop configuration (easily toggled for quick tests)
        self.early_stop_enabled = True
        self.early_stop_min_cycles = 5
        self.early_stop_ges_threshold = 100
        self.early_stop_force = False
        self.early_stop_triggered_once = False
        
                # NEW: Skip pretraining once (set by controller after threshold)
        self.skip_next_pretraining = False
        
        # NEW: Toggle to switch addition strategy after threshold
        self.intelligent_addition_enabled = False
        
        # NEW: Permanently skip all future pretraining once threshold hit
        self.skip_pretraining_always = False
        
        # NEW: Phase-change indicator configuration
        self.phase_change_indicator = phase_change_indicator  # 'ges' or 'novel_test'
        self.novel_test_interval_cycles = novel_test_interval_cycles
        self.novel_test_threshold_correct = novel_test_threshold_correct
        self.novel_test_bidirectional = novel_test_bidirectional
        self.novel_test_log_summary_only = novel_test_log_summary_only
        
        # NEW: Distractor scheduling state (based on first GES-threshold hit)
        self.ges_threshold_hit_phase = None
        self.base_num_distractors_at_threshold = None

        # === NEW: Background decoder training state ===
        self.decoder_background_enabled = False
        # Reconstruction sample logging cadence
        self._recon_step_counter = 0
        self.recon_sample_interval = 10
        # Selection sample logging cadence (for input-output display)
        self._selection_step_counter = 0
        self.selection_sample_interval = 10
        self.last_selection_sample = None

    def set_puzzle_dataset(self, puzzles: List[Puzzle]):
        """Set the full ARC puzzle dataset"""
        self.available_arc_puzzles = puzzles
        print(f"Loaded {len(puzzles)} total ARC puzzles for iterative training")
    
    def initialize_first_puzzles(self):
        """
        Initialize the first set of active puzzles with RANDOM selection.
        Modified to randomly sample instead of taking first N puzzles.
        """
        if len(self.available_arc_puzzles) < self.initial_puzzle_count:
            raise ValueError(f"Need at least {self.initial_puzzle_count} puzzles to start")
        
        # CHANGED: Randomly sample initial puzzles instead of taking first N
        available_indices = list(range(len(self.available_arc_puzzles)))
        selected_indices = random.sample(available_indices, self.initial_puzzle_count)
        selected_indices.sort()  # Sort for consistent ordering in active_puzzles
        
        # Track used puzzle indices
        self.used_puzzle_indices.update(selected_indices)
        
        # Get the randomly selected puzzles
        self.active_puzzles = [self.available_arc_puzzles[i] for i in selected_indices]
        
        # Create symbol assignments starting AFTER any frozen symbols
        self.puzzle_symbol_mapping = {}
        self.symbol_puzzle_mapping = {}
        
        # Account for frozen symbols from previous phases
        # When resuming from snapshot, agents already have frozen_comm_symbols set
        frozen_comm_symbols = self.agent1.current_comm_symbols if hasattr(self, 'frozen_comm_symbols') and self.frozen_comm_symbols > 0 else 0
        start_symbol = self.agent1.puzzle_symbols + frozen_comm_symbols
        
        print(f"Starting symbol assignment at index {start_symbol}")
        if frozen_comm_symbols > 0:
            print(f"  (accounting for {frozen_comm_symbols} frozen symbols: {self.agent1.puzzle_symbols}-{self.agent1.puzzle_symbols + frozen_comm_symbols - 1})")
        
        for i, puzzle in enumerate(self.active_puzzles):
            symbol_idx = start_symbol + i
            self.puzzle_symbol_mapping[i] = symbol_idx
            self.symbol_puzzle_mapping[symbol_idx] = i
        
        self.next_available_symbol = start_symbol + len(self.active_puzzles)
        
        # Update agent vocabularies: frozen symbols (already set) + new symbols
        new_comm_symbols = len(self.active_puzzles)
        total_comm_symbols = frozen_comm_symbols + new_comm_symbols
        self.agent1.current_comm_symbols = total_comm_symbols
        self.agent2.current_comm_symbols = total_comm_symbols
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + total_comm_symbols
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + total_comm_symbols
        
        # Assign symbols to the current unfrozen position
        # Determine which position we're training based on frozen positions
        frozen_positions = getattr(self, 'frozen_positions', [])
        if frozen_positions:
            # Next unfrozen position is after the highest frozen position
            current_training_position = max(frozen_positions) + 1
        else:
            # No frozen positions, training position 0
            current_training_position = 0
        
        # Get the symbol indices we just created
        new_symbol_indices = set(self.puzzle_symbol_mapping.values())
        
        # Assign these symbols ONLY to the current training position
        self.agent1.set_position_vocabulary(current_training_position, new_symbol_indices)
        self.agent2.set_position_vocabulary(current_training_position, new_symbol_indices)
        
        print(f"Initialized with {len(self.active_puzzles)} RANDOMLY SELECTED puzzles")
        print(f"Selected puzzle indices: {selected_indices}")
        print(f"Symbol assignments: {self.puzzle_symbol_mapping}")
        print(f"Next available symbol: {self.next_available_symbol}")
        if frozen_comm_symbols > 0:
            print(f"Frozen symbols: {frozen_comm_symbols} (indices {self.agent1.puzzle_symbols}-{self.agent1.puzzle_symbols + frozen_comm_symbols - 1})")
            print(f"New symbols: {new_comm_symbols} (indices {start_symbol}-{start_symbol + new_comm_symbols - 1})")
            print(f"Total communication symbols: {frozen_comm_symbols} + {new_comm_symbols} = {total_comm_symbols}")
        else:
            print(f"Communication symbols: {total_comm_symbols} (no frozen symbols)")
        print(f"Assigned symbols {sorted(new_symbol_indices)} to position {current_training_position}")

    
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
        """Generate random noise grids as distractors"""
        distractors = []
        
        # Generate num_distractors random noise grids
        for _ in range(self.num_distractors):
            # Random grid size between 1x1 and 30x30
            height = random.randint(1, 30)
            width = random.randint(1, 30)
            
            # Fill with random puzzle symbols (0-9)
            random_grid = np.random.randint(0, 10, size=(height, width))
            
            # Convert to tensor with shape [1, H, W]
            distractor_tensor = torch.tensor(
                random_grid,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            distractors.append(distractor_tensor)
        
        return distractors

    def run_consolidation_test(self) -> Dict[str, list]:
        """
        Run consolidation tests to identify recessive symbols.
        Robust version:
        • Runs 10 selection tests per mapped puzzle
        • Records full candidate info (grid + confidence + selected/target flags)
        • Saves details keyed by BOTH puzzle_idx and symbol to avoid lookup gaps
        Returns:
        confusion_data: {symbol: [symbol or -1 for each test]}
        """
        from collections import defaultdict

        if not self.web_mode:
            print(f"\n{'='*50}")
            print(f"CONSOLIDATION PHASE - Testing Symbol Accuracy")
            print(f"Modified: 10 selection tests per puzzle with {self.num_distractors} distractors each")
            print(f"{'='*50}")
        else:
            print(f"\nConsolidation phase: testing {len(list(self.puzzle_symbol_mapping.keys()))} symbols...")

        self.agent1.eval()
        self.agent2.eval()

        # For accuracies → used by identify_recessive_symbols
        puzzle_test_results = defaultdict(list)   # {puzzle_idx: [bool, ...]}

        # For visual debugging → now ALWAYS populated per test
        by_puzzle = defaultdict(list)             # {puzzle_idx: [(test_num, all_candidates_data), ...]}
        by_symbol = defaultdict(list)             # {symbol:     [(test_num, all_candidates_data), ...]}

        with torch.no_grad():
            mapped_puzzle_indices = list(self.puzzle_symbol_mapping.keys())
            if not self.web_mode:
                print(f"Testing {len(mapped_puzzle_indices)} puzzles with symbol mappings")

            for puzzle_idx in mapped_puzzle_indices:
                puzzle = self.active_puzzles[puzzle_idx]
                # INPUT tensor for encoding
                input_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                # OUTPUT tensor for selection
                output_tensor = torch.tensor(puzzle.test_output, dtype=torch.long, device=self.device).unsqueeze(0)

                symbol = self.puzzle_symbol_mapping[puzzle_idx]
                if not self.web_mode:
                    print(f"\nTesting Puzzle {puzzle_idx} (Symbol {symbol}) - 10 selection tests:")

                correct = 0
                for test_num in range(10):
                    # Sender encodes INPUT → message
                    symbols, _, _ = self.agent1.encode_puzzle_to_message(
                        input_tensor, temperature=0.1, deterministic=True
                    )

                    # Build OUTPUT candidates: target OUTPUT first, then distractor OUTPUTs
                    distractors = self.sample_distractors(output_tensor, puzzle_idx)
                    candidates = [output_tensor] + distractors  # each [1, H, W]

                    # Receiver selects from OUTPUT candidates
                    selection_probs, selection_logits, _ = self.agent2.select_from_candidates(
                        symbols, candidates, temperature=0.1
                    )
                    # predicted candidate index
                    predicted_idx = int(selection_logits.argmax(dim=-1).item())
                    is_correct = (predicted_idx == 0)

                    # Save correctness for confusion matrix later
                    puzzle_test_results[puzzle_idx].append(is_correct)
                    if is_correct:
                        correct += 1

                    # Build full candidate metadata (ALWAYS, not only on errors)
                    all_candidates_data = []
                    for cand_idx, cand in enumerate(candidates):
                        cand_np = cand[0].detach().cpu().numpy()  # remove batch
                        conf = float(selection_probs[0, cand_idx].item())
                        all_candidates_data.append({
                            "puzzle": cand_np,
                            "confidence": conf,
                            "is_target": (cand_idx == 0),
                            "is_selected": (cand_idx == predicted_idx),
                            "candidate_idx": cand_idx,
                        })

                    # Store under both keys so later lookups are stable
                    record = (test_num, all_candidates_data)
                    by_puzzle[puzzle_idx].append(record)
                    by_symbol[symbol].append(record)

                    # A little progress print
                    if not self.web_mode and (test_num < 3 or test_num == 9):
                        status = "✓" if is_correct else "✗"
                        tgt_conf = float(selection_probs[0, 0].item())
                        print(f"    Test {test_num+1}/10: Selected candidate {predicted_idx} {status} (target conf: {tgt_conf:.3f})")

                acc = correct / 10
                if not self.web_mode:
                    print(f"  → Final accuracy: {correct}/10 ({acc:.1%})")
                    if correct == 0:
                        print(f"  → WARNING: Puzzle {puzzle_idx} was NEVER selected correctly!")
                elif correct == 0:
                    # Still show critical warnings in web mode
                    print(f"  WARNING: Symbol {symbol} never selected correctly!")

        # Build confusion_data in the format expected by identify_recessive_symbols
        confusion_data = {}
        for puzzle_idx, results in puzzle_test_results.items():
            symbol = self.puzzle_symbol_mapping[puzzle_idx]
            confusion_data[symbol] = [symbol if ok else -1 for ok in results]

        # Make the test details available for the analyzer
        self.puzzle_test_details_by_puzzle = dict(by_puzzle)
        self.puzzle_test_details_by_symbol = dict(by_symbol)

        return confusion_data


    def print_puzzle_grid(self, grid: np.ndarray, title: str = "Grid", indent: str = "    "):
        """
        Print a puzzle grid in a readable format with proper indentation.
        
        Args:
            grid: 2D numpy array representing the puzzle
            title: Title to display above the grid
            indent: Indentation string for formatting
        """
        print(f"{indent}{title}:")
        for row in grid:
            print(f"{indent}  " + " ".join(f"{x:2d}" for x in row))

    def identify_recessive_symbols(self, confusion_data: Dict[int, List[int]]) -> Set[int]:
        """
        Identify symbols that are consistently misinterpreted.
        Modified to remove symbols with accuracy ≤ consolidation_threshold (configurable, default 30%).
        Shows visual debugging for poor performers, with robust fallbacks.
        Also writes all output to consolidation_analysis.txt.
        
        IMPORTANT: Frozen symbols (from previous training phases) are never removed.
        """
        recessive_symbols: Set[int] = set()
        poor_performers: List[int] = []  # Symbols with ≤ threshold accuracy (but >0%)

        consolidation_filename = 'consolidation_analysis.txt'
        threshold_percent = self.consolidation_threshold * 100
        
        # Determine frozen symbol range
        frozen_start = self.agent1.puzzle_symbols
        frozen_end = frozen_start + getattr(self, 'frozen_comm_symbols', 0)

        if not self.web_mode:
            header = f"{'='*50}\nANALYZING SYMBOL PERFORMANCE (10 tests per symbol)\nThreshold: {threshold_percent:.0f}%\n{'='*50}"
            print(header)
            if frozen_end > frozen_start:
                print(f"Protected frozen symbols: {frozen_start}-{frozen_end-1} (will not be removed)")
        else:
            header = f"Analyzing symbol performance (threshold: {threshold_percent:.0f}%)..."
            print(header)
        
        with open(consolidation_filename, 'a') as log_file:
            log_file.write(f"\n{header}\n")
            if frozen_end > frozen_start:
                log_file.write(f"Protected frozen symbols: {frozen_start}-{frozen_end-1}\n")

        # --- Compute per-symbol accuracy and categorize ---
        for symbol, predictions in confusion_data.items():
            # Skip frozen symbols - they should never be removed
            if frozen_start <= symbol < frozen_end:
                if not self.web_mode:
                    print(f"Symbol {symbol}: FROZEN (skipping consolidation analysis)")
                continue
            total_tests = len(predictions)
            correct_predictions = sum(1 for pred in predictions if pred == symbol)
            accuracy = (correct_predictions / total_tests) if total_tests > 0 else 0.0

            symbol_line = f"Symbol {symbol}: {correct_predictions}/{total_tests} tests correct ({accuracy:.1%})"
            if not self.web_mode:
                print(symbol_line)
            with open(consolidation_filename, 'a') as log_file:
                log_file.write(f"{symbol_line}\n")

            # CHANGED: Now include all symbols with ≤ threshold accuracy for removal
            if accuracy <= self.consolidation_threshold:
                recessive_symbols.add(symbol)
                if accuracy == 0.0:
                    status_line = f"  → RECESSIVE: Symbol {symbol} never selected correctly in any test"
                else:
                    status_line = f"  → RECESSIVE: Symbol {symbol} selected correctly ≤{threshold_percent:.0f}% of the time"
            elif accuracy < 0.50:
                status_line = f"  → LOW PERFORMANCE: Symbol {symbol} selected correctly <50% of the time"
            else:
                status_line = f"  → GOOD PERFORMANCE: Symbol {symbol} performing well"

            if not self.web_mode:
                print(status_line)
            with open(consolidation_filename, 'a') as log_file:
                log_file.write(f"{status_line}\n")

        if not self.web_mode:
            summary_text = (
                f"\nSummary:\n"
                f"  Total symbols tested: {len(confusion_data)}\n"
                f"  Recessive symbols (≤30% accuracy): {len(recessive_symbols)}\n"
                f"  Symbols to be removed: {sorted(recessive_symbols) if recessive_symbols else 'None'}"
            )
            print(summary_text)
        else:
            # Simplified summary for web mode
            if recessive_symbols:
                print(f"Found {len(recessive_symbols)} recessive symbols to remove: {sorted(recessive_symbols)}")
            else:
                print("No recessive symbols found - all symbols performing well")
        
        # Always write full summary to file
        summary_text = (
            f"\nSummary:\n"
            f"  Total symbols tested: {len(confusion_data)}\n"
            f"  Recessive symbols (≤30% accuracy): {len(recessive_symbols)}\n"
            f"  Symbols to be removed: {sorted(recessive_symbols) if recessive_symbols else 'None'}"
        )
        with open(consolidation_filename, 'a') as log_file:
            log_file.write(f"{summary_text}\n")

        # --- Visual debugging for recessive symbols (now all ≤30% performers) ---
        if recessive_symbols and not self.web_mode:
            debug_header = (
                f"\n{'='*70}\n"
                f"VISUAL DEBUG: WRONG ANSWERS FOR RECESSIVE SYMBOLS (≤30% accuracy)\n"
                f"{'='*70}"
            )
            print(debug_header)
            with open(consolidation_filename, 'a') as log_file:
                log_file.write(f"{debug_header}\n")

                # Sort symbols to debug
                symbols_to_debug = sorted(recessive_symbols)

                for symbol in symbols_to_debug:
                    # Find the active puzzle index for this symbol
                    puzzle_idx = None
                    for p_idx, s in self.puzzle_symbol_mapping.items():
                        if s == symbol:
                            puzzle_idx = p_idx
                            break

                    if puzzle_idx is None:
                        warn = f"Warning: Could not find puzzle for symbol {symbol}"
                        print(warn)
                        log_file.write(warn + "\n")
                        continue

                    # Accuracy recap
                    predictions = confusion_data.get(symbol, [])
                    total_tests = len(predictions)
                    correct_predictions = sum(1 for pred in predictions if pred == symbol)
                    accuracy = (correct_predictions / total_tests) if total_tests > 0 else 0.0

                    header_line = (
                        f"\nSymbol {symbol} (Puzzle {puzzle_idx}) - Accuracy: {accuracy:.1%}\n"
                        f"Showing all {max(total_tests - correct_predictions, 0)} wrong answers:"
                    )
                    print(header_line)
                    log_file.write(header_line + "\n")

                    # Retrieve recorded test details (prefer new robust stores, fallback to legacy)
                    details = []
                    if hasattr(self, "puzzle_test_details_by_puzzle"):
                        details = list(self.puzzle_test_details_by_puzzle.get(puzzle_idx, []))
                    if not details and hasattr(self, "puzzle_test_details_by_symbol"):
                        details = list(self.puzzle_test_details_by_symbol.get(symbol, []))

                    # Legacy fallback: self.puzzle_wrong_answers[puzzle_idx] may contain only wrong tests.
                    if not details and hasattr(self, "puzzle_wrong_answers"):
                        legacy = self.puzzle_wrong_answers.get(puzzle_idx, [])
                        # Normalize legacy tuples into the new (test_num, all_candidates_data) shape
                        normalized = []
                        for item in legacy:
                            if isinstance(item, (list, tuple)):
                                if len(item) == 2:
                                    # Already new form: (test_num, all_candidates_data)
                                    test_num, all_candidates_data = item
                                    normalized.append((test_num, all_candidates_data))
                                elif len(item) == 3:
                                    # Old form: (test_num, target_puzzle, selected_puzzle)
                                    test_num, target_puzzle, selected_puzzle = item
                                    all_candidates_data = [
                                        {
                                            "puzzle": target_puzzle,
                                            "confidence": 0.0,
                                            "is_target": True,
                                            "is_selected": False,
                                            "candidate_idx": 0,
                                        },
                                        {
                                            "puzzle": selected_puzzle,
                                            "confidence": 0.0,
                                            "is_target": False,
                                            "is_selected": True,
                                            "candidate_idx": 1,
                                        },
                                    ]
                                    normalized.append((test_num, all_candidates_data))
                                # else: unexpected shape → ignore
                        details = normalized

                    # If we still have no details, emit a clear diagnostic and continue
                    if not details:
                        no_data = (
                            f"  No wrong answer data available for puzzle {puzzle_idx} / symbol {symbol} "
                            f"(check test recording)."
                        )
                        print(no_data)
                        log_file.write(no_data + "\n")
                        continue

                    # Filter to only the tests where the target was NOT selected
                    wrong_only = []
                    for test_num, cand_list in details:
                        target_selected = any(c.get("is_target") and c.get("is_selected") for c in cand_list)
                        if not target_selected:
                            wrong_only.append((test_num, cand_list))

                    if not wrong_only:
                        info = "  (No wrong picks recorded – all tests selected the target.)"
                        print(info)
                        log_file.write(info + "\n")
                        continue

                    # Emit detailed diagnostics for each wrong test
                    for i, (test_num, cand_list) in enumerate(wrong_only, 1):
                        # Confidence ranking
                        ranked = sorted(cand_list, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                        ranking_info = "    Confidence ranking: " + " > ".join(
                            f"Cand{c.get('candidate_idx', -1)}({'T' if c.get('is_target') else 'D'}): "
                            f"{float(c.get('confidence', 0.0)):.3f}" for c in ranked
                        )
                        print(ranking_info)
                        log_file.write(ranking_info + "\n")

                        # Summary (selected vs target)
                        sel = next((c for c in cand_list if c.get("is_selected")), None)
                        tgt = next((c for c in cand_list if c.get("is_target")), None)
                        if sel is not None and tgt is not None:
                            summary = (
                                f"    Summary: Agent selected candidate {sel.get('candidate_idx', -1)} "
                                f"(conf: {float(sel.get('confidence', 0.0)):.3f}) instead of target "
                                f"(conf: {float(tgt.get('confidence', 0.0)):.3f})"
                            )
                            print(summary)
                            log_file.write(summary + "\n")

                        # Show each candidate block with grid
                        for c in cand_list:
                            is_target = bool(c.get("is_target"))
                            is_selected = bool(c.get("is_selected"))
                            cand_idx = c.get("candidate_idx", -1)
                            conf_val = float(c.get("confidence", 0.0))

                            if is_target and is_selected:
                                title, mark = f"CANDIDATE {cand_idx}: TARGET (CORRECTLY SELECTED)", "✓✓"
                            elif is_target and not is_selected:
                                title, mark = f"CANDIDATE {cand_idx}: TARGET (NOT SELECTED)", "✗✗"
                            elif (not is_target) and is_selected:
                                title, mark = f"CANDIDATE {cand_idx}: DISTRACTOR (WRONGLY SELECTED)", "✗"
                            else:
                                title, mark = f"CANDIDATE {cand_idx}: DISTRACTOR (NOT SELECTED)", " "

                            conf_line = f"      {mark} Confidence: {conf_val:.3f}"
                            print(conf_line)
                            log_file.write(conf_line + "\n")

                            # Render grids to console and file
                            self.print_puzzle_grid(c.get("puzzle"), title=f"      {title}", indent="")
                            self.write_puzzle_grid_to_file(log_file, c.get("puzzle"), f"      {title}", "")

                        log_file.write("\n")

        # Completion marker
        with open(consolidation_filename, 'a') as log_file:
            log_file.write(f"\n{'='*70}\nCONSOLIDATION ANALYSIS COMPLETE\n{'='*70}\n\n")

        return recessive_symbols


    def write_puzzle_grid_to_file(self, log_file, grid: np.ndarray, title: str = "Grid", indent: str = "    "):
        """
        Write a puzzle grid to the log file in the same format as console output.
        
        Args:
            log_file: Open file handle to write to
            grid: 2D numpy array representing the puzzle
            title: Title to display above the grid
            indent: Indentation string for formatting
        """
        log_file.write(f"{indent}{title}:\n")
        for row in grid:
            log_file.write(f"{indent}  " + " ".join(f"{x:2d}" for x in row) + "\n")

    def _get_timestamp(self):
        """Get current timestamp for logging"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def remove_recessive_symbols(self, recessive_symbols: Set[int]):
        """Remove recessive symbols but keep their associated puzzles"""
        if not recessive_symbols:
            if not self.web_mode:
                print("No recessive symbols to remove")
            return
        
        if not self.web_mode:
            print(f"\n{'='*50}")
            print(f"REMOVING RECESSIVE SYMBOLS: {recessive_symbols}")
            print(f"{'='*50}")
        else:
            print(f"Removing {len(recessive_symbols)} recessive symbols...")
        
        # Find puzzles that will lose their symbol mappings
        orphaned_puzzles = []
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                orphaned_puzzles.append(puzzle_idx)
        
        if not self.web_mode:
            print(f"Puzzles that will lose symbol mappings: {orphaned_puzzles}")
            print(f"These puzzles will remain active but won't have dedicated symbols")
        
        # STORE ORIGINAL MAPPINGS BEFORE REMOVAL
        original_puzzle_symbol_mapping = self.puzzle_symbol_mapping.copy()
        
        # Remove symbol mappings for recessive symbols
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                if not self.web_mode:
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
                if not self.web_mode:
                    print(f"Symbol transfer: {old_symbol} -> {new_symbol} (puzzle {puzzle_idx})")
            
            new_symbol_idx += 1
        
        # UPDATE MAPPINGS
        self.puzzle_symbol_mapping = new_puzzle_mapping
        self.symbol_puzzle_mapping = new_symbol_mapping
        
        # TRANSFER EMBEDDINGS FOR REMAPPED SYMBOLS
        if symbol_transfer_mapping:
            if not self.web_mode:
                print(f"\nTransferring embeddings for {len(symbol_transfer_mapping)} remapped symbols...")
            
            with torch.no_grad():
                # Transfer embeddings for Agent 1
                for old_symbol, new_symbol in symbol_transfer_mapping.items():
                    self.agent1.communication_embedding.weight[new_symbol].copy_(
                        self.agent1.communication_embedding.weight[old_symbol]
                    )
                    if not self.web_mode:
                        print(f"  Agent1: Copied embedding {old_symbol} -> {new_symbol}")
                
                # Transfer embeddings for Agent 2
                for old_symbol, new_symbol in symbol_transfer_mapping.items():
                    self.agent2.communication_embedding.weight[new_symbol].copy_(
                        self.agent2.communication_embedding.weight[old_symbol]
                    )
                    if not self.web_mode:
                        print(f"  Agent2: Copied embedding {old_symbol} -> {new_symbol}")
            
            if not self.web_mode:
                print(f"Embedding transfer complete: {len(symbol_transfer_mapping)} transfers")
        else:
            if not self.web_mode:
                print("No embedding transfers needed (symbols remain in same positions)")
        
        # Update agent vocabularies based on remaining mapped symbols
        self.agent1.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent2.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + len(self.puzzle_symbol_mapping)
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + len(self.puzzle_symbol_mapping)
        
        # Track removed symbols
        self.removed_symbols.update(recessive_symbols)
        
        if not self.web_mode:
            print(f"\nResults:")
            print(f"  Total puzzles (unchanged): {len(self.active_puzzles)}")
            print(f"  Puzzles with symbol mappings: {len(self.puzzle_symbol_mapping)}")
            print(f"  Puzzles without symbol mappings: {len(self.active_puzzles) - len(self.puzzle_symbol_mapping)}")
            print(f"  Active communication symbols: {self.agent1.current_comm_symbols}")
            print(f"  New symbol mapping: {self.puzzle_symbol_mapping}")
            print(f"  Embeddings preserved for surviving symbols: ✓")
        else:
            print(f"Consolidation complete: {self.agent1.current_comm_symbols} active symbols remaining")
    
    def add_new_puzzles(self):
        """
        Add new puzzles from the ARC dataset using RANDOM selection.
        Behavior:
        - Before threshold: assign NEW symbols using the next contiguous index (random embeddings).
        - After threshold: allocate NEW symbols at encoder-predicted embeddings (no pretraining).
        """
        if not self.web_mode:
            print(f"\n{'='*50}")
            if self.intelligent_addition_enabled:
                print(f"ADDITION PHASE - Adding {self.puzzles_per_addition} new puzzles (predicted-embedding symbols)")
            else:
                print(f"ADDITION PHASE - Adding {self.puzzles_per_addition} new puzzles (random symbol initialization)")
            print(f"{'='*50}")
        else:
            print(f"Adding {self.puzzles_per_addition} new puzzles...")
        
        # Find unused puzzle indices
        all_indices = set(range(len(self.available_arc_puzzles)))
        unused_indices = list(all_indices - self.used_puzzle_indices)
        
        if len(unused_indices) < self.puzzles_per_addition:
            actual_addition = len(unused_indices)
            print(f"Only {actual_addition} unused puzzles available to add")
            self.puzzles_per_addition = actual_addition
        
        if self.puzzles_per_addition <= 0:
            print("No more unused puzzles to add!")
            return []
        
        # Randomly sample from unused puzzles
        selected_new_indices = random.sample(unused_indices, self.puzzles_per_addition)
        selected_new_indices.sort()  # Sort for consistent ordering
        
        # Track the newly used indices
        self.used_puzzle_indices.update(selected_new_indices)
        
        new_puzzles = []
        
        for dataset_idx in selected_new_indices:
            new_puzzle = self.available_arc_puzzles[dataset_idx]
            self.active_puzzles.append(new_puzzle)
            new_puzzles.append(new_puzzle)
            
            # Assign active puzzle index
            active_puzzle_idx = len(self.active_puzzles) - 1
            
            if self.intelligent_addition_enabled:
                # Predict and allocate symbol at predicted embedding
                target_tensor = torch.tensor(new_puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                seq_emb = self.agent1.embedding_system.embed_puzzle(target_tensor)  # [1, L, D]
                pred_emb = self.agent1.encoder.predict_symbol_embedding(seq_emb, position=0)  # [1, D]
                pred_emb = F.normalize(pred_emb, p=2, dim=-1)
                new_sym_idx_1 = self.agent1.add_new_symbol_with_embedding(pred_emb)
                self.agent2.add_new_symbol_with_embedding(pred_emb)
                self.puzzle_symbol_mapping[active_puzzle_idx] = new_sym_idx_1
                self.symbol_puzzle_mapping[new_sym_idx_1] = active_puzzle_idx
                if not self.web_mode:
                    print(f"Added dataset puzzle {dataset_idx} as active puzzle {active_puzzle_idx} with symbol {new_sym_idx_1} (predicted embedding)")
            else:
                # Random symbol initialization: assign next contiguous symbol index
                next_symbol = self.agent1.puzzle_symbols + len(self.puzzle_symbol_mapping)
                self.puzzle_symbol_mapping[active_puzzle_idx] = next_symbol
                self.symbol_puzzle_mapping[next_symbol] = active_puzzle_idx
                if not self.web_mode:
                    print(f"Added dataset puzzle {dataset_idx} as active puzzle {active_puzzle_idx} with symbol {next_symbol} (random initialization)")
         
        # Update agent vocabularies to reflect the number of mapped puzzles
        self.agent1.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent2.current_comm_symbols = len(self.puzzle_symbol_mapping)
        self.agent1.current_total_symbols = self.agent1.puzzle_symbols + len(self.puzzle_symbol_mapping)
        self.agent2.current_total_symbols = self.agent2.puzzle_symbols + len(self.puzzle_symbol_mapping)
        
        # Assign new symbols to the current unfrozen position
        # Determine which position we're training based on frozen positions
        frozen_positions = getattr(self, 'frozen_positions', [])
        if frozen_positions:
            # Next unfrozen position is after the highest frozen position
            current_training_position = max(frozen_positions) + 1
        else:
            # No frozen positions, training position 0
            current_training_position = 0
        
        # Get ONLY the newly added symbol indices
        new_symbol_indices = set()
        for active_idx in range(len(self.active_puzzles) - len(new_puzzles), len(self.active_puzzles)):
            if active_idx in self.puzzle_symbol_mapping:
                new_symbol_indices.add(self.puzzle_symbol_mapping[active_idx])
        
        # Add these new symbols to the current position's vocabulary
        if new_symbol_indices:
            existing_vocab_1 = self.agent1.position_vocabularies.get(current_training_position, set())
            existing_vocab_2 = self.agent2.position_vocabularies.get(current_training_position, set())
            
            self.agent1.set_position_vocabulary(current_training_position, existing_vocab_1 | new_symbol_indices)
            self.agent2.set_position_vocabulary(current_training_position, existing_vocab_2 | new_symbol_indices)
            
            if not self.web_mode:
                print(f"Added symbols {sorted(new_symbol_indices)} to position {current_training_position}")
        
        if not self.web_mode:
            print(f"Total active puzzles: {len(self.active_puzzles)}")
            print(f"Selected new puzzle indices from dataset: {selected_new_indices}")
            print(f"Total used puzzle indices: {len(self.used_puzzle_indices)}")
            print(f"Remaining unused puzzles: {len(self.available_arc_puzzles) - len(self.used_puzzle_indices)}")
            print(f"New symbol mapping: {self.puzzle_symbol_mapping}")
        else:
            print(f"Added {len(new_puzzles)} puzzles. Total: {len(self.active_puzzles)} active puzzles, {self.agent1.current_comm_symbols} symbols")
        
        return new_puzzles
    
    # --- NEW: Evaluation helpers for remedial training ---
    def evaluate_selection_accuracy(self, puzzle_indices: List[int] = None, tests: int = 10, temperature: float = 0.1) -> Dict[int, int]:
        """Evaluate selection accuracy (Agent1 encodes INPUT → Agent2 selects OUTPUT) for given puzzles.
        Returns a dict of puzzle_idx -> number of correct selections out of `tests`.
        """
        self.agent1.eval()
        self.agent2.eval()
        results: Dict[int, int] = {}
        indices = puzzle_indices if puzzle_indices is not None else list(range(len(self.active_puzzles)))
        with torch.no_grad():
            for puzzle_idx in indices:
                puzzle = self.active_puzzles[puzzle_idx]
                # INPUT tensor for encoding
                input_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                # OUTPUT tensor for selection
                output_tensor = torch.tensor(puzzle.test_output, dtype=torch.long, device=self.device).unsqueeze(0)
                correct = 0
                for _ in range(tests):
                    # Encode INPUT
                    symbols, _, _ = self.agent1.encode_puzzle_to_message(
                        input_tensor, temperature=temperature, deterministic=True
                    )
                    # Build OUTPUT candidates
                    distractors = self.sample_distractors(output_tensor, puzzle_idx)
                    candidates = [output_tensor] + distractors
                    # Select from OUTPUT candidates
                    _, selection_logits, _ = self.agent2.select_from_candidates(
                        symbols, candidates, temperature=temperature
                    )
                    predicted_idx = int(selection_logits.argmax(dim=-1).item())
                    if predicted_idx == 0:
                        correct += 1
                results[puzzle_idx] = correct
        self.agent1.train()
        self.agent2.train()
        return results
    

    def get_weak_puzzles(self, accuracy_threshold: float = 0.7, tests: int = 10) -> List[int]:
        """Return indices of puzzles whose selection accuracy is below threshold."""
        threshold_correct = int(np.ceil(accuracy_threshold * tests))
        eval_results = self.evaluate_selection_accuracy(tests=tests)
        weak = [idx for idx, correct in eval_results.items() if correct < threshold_correct]
        print(f"Identified {len(weak)} weak puzzles (<{threshold_correct}/{tests} correct): {weak}")
        return weak
    
    def advance_phase(self):
        """Advance to the next phase in the cycle"""
        if self.current_phase == "pretraining":
            self.current_phase = "training"
        elif self.current_phase == "training":
            # self.current_phase = "addition"
            self.current_phase = "consolidation"
        elif self.current_phase == "consolidation":
            # NEW: Insert remedial phase between consolidation and addition
            self.current_phase = "addition" # can be switched with remedial
        elif self.current_phase == "remedial":
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
        
        # NEW: Update distractor scheduling after any phase change (uses current global_phase_count)
        self._update_distractors_for_current_phase()

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
        
        # If reconstruction mode is active, run reconstruction training instead of selection
        if getattr(self, 'reconstruction_mode', False):
            return [self._train_reconstruction_step(puzzle, puzzle_idx, temperature=temperature)]

        metrics_history = []
        
        # Set training mode based on phase
        if self.current_phase == "pretraining":
            self.set_training_mode("encoder_only")
        else:
            self.set_training_mode("joint")
        
        for exchange in range(num_exchanges):
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            
            # Get input and output tensors from puzzle
            # puzzle parameter is the INPUT tensor (kept for backward compatibility)
            input_tensor = puzzle
            
            # Get the output tensor from the active puzzle object
            puzzle_obj = self.active_puzzles[puzzle_idx]
            output_tensor = torch.tensor(
                puzzle_obj.test_output,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            
            # Sample distractors for this exchange (output puzzles)
            distractors = self.sample_distractors(output_tensor, puzzle_idx)
            
            # Agent1 encodes INPUT, Agent2 selects OUTPUT
            symbols1, symbol_logits1, length_stats1 = self.agent1.encode_puzzle_to_message(
                input_tensor, temperature=temperature, initial_phase=initial_phase
            )
            
            # Prepare candidates: target OUTPUT + distractor OUTPUTs
            candidates1 = [output_tensor] + distractors  # each [1, H, W]
            
            # Receiver selects from OUTPUT candidates
            selection_probs1, selection_logits1, debug_info1 = self.agent2.select_from_candidates(
                symbols1, candidates1, temperature=temperature
            )
            
            # Agent2 encodes INPUT, Agent1 selects OUTPUT (bidirectional)
            symbols2, symbol_logits2, length_stats2 = self.agent2.encode_puzzle_to_message(
                input_tensor, temperature=temperature, initial_phase=initial_phase
            )
            
            # Sample different distractors for reverse direction (output puzzles)
            distractors2 = self.sample_distractors(output_tensor, puzzle_idx)
            candidates2 = [output_tensor] + distractors2
            
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
            
            # Emit a lightweight reconstruction sample intermittently even in selection mode
            # This allows the UI to always display latest reconstructions while background decoding trains
            self._recon_step_counter += 1
            try:
                if self._recon_step_counter % max(1, int(self.recon_sample_interval)) == 0:
                    with torch.no_grad():
                        # Target is the OUTPUT tensor
                        target_tensor = output_tensor
                        # Extract all symbols from Agent1's message (for multi-position messages)
                        seq_len = symbols1.size(1) if symbols1.dim() == 3 else 1
                        message_symbols_local = []
                        message_symbols_abs = []
                        
                        for pos in range(seq_len):
                            if symbols1.dim() == 3:
                                local_sym = int(symbols1[0, pos].argmax().item())
                            else:
                                local_sym = 0
                            abs_sym = int(self.agent1.puzzle_symbols + local_sym)
                            message_symbols_local.append(local_sym)
                            message_symbols_abs.append(abs_sym)
                        
                        # Decode using Agent2's decoder (should reconstruct OUTPUT from message)
                        comm_emb_a2 = self.agent2.communication_embedding.weight[
                            self.agent2.puzzle_symbols:self.agent2.current_total_symbols
                        ]
                        embedded_msg1 = torch.matmul(symbols1, comm_emb_a2)
                        # Pass input_tensor to decoder so it can start from the input grid
                        logits_sel, _, _, _ = self.agent2.decoder(embedded_msg1, temperature=1.0, input_grid=input_tensor)
                        Bp, Hp, Wp, Cp = logits_sel.shape
                        Ht, Wt = int(target_tensor.shape[1]), int(target_tensor.shape[2])
                        # Don't crop - show full grids at their actual sizes
                        tgt_np = target_tensor[0].detach().cpu().long().numpy().tolist()
                        pred_np = logits_sel[0].argmax(dim=-1).detach().cpu().long().numpy().tolist()
                        metrics['recon_sample'] = {
                            'direction': 'A1_to_A2',
                            'message_symbols_local': message_symbols_local,
                            'message_symbols_abs': message_symbols_abs,
                            # Keep backward compatibility
                            'message_symbol_local': message_symbols_local[0] if message_symbols_local else 0,
                            'message_symbol_abs': message_symbols_abs[0] if message_symbols_abs else 0,
                            'sequence_length': seq_len,
                            'target': tgt_np,
                            'reconstruction': pred_np,
                            'target_size': [Ht, Wt],
                            'predicted_size': [Hp, Wp]
                        }
            except Exception:
                pass
            
            # Emit selection sample intermittently to show input-output task
            self._selection_step_counter += 1
            try:
                if self._selection_step_counter % max(1, int(self.selection_sample_interval)) == 0:
                    with torch.no_grad():
                        # Extract message symbols
                        seq_len = symbols1.size(1) if symbols1.dim() == 3 else 1
                        message_symbols_local = []
                        message_symbols_abs = []
                        
                        for pos in range(seq_len):
                            if symbols1.dim() == 3:
                                local_sym = int(symbols1[0, pos].argmax().item())
                            else:
                                local_sym = 0
                            abs_sym = int(self.agent1.puzzle_symbols + local_sym)
                            message_symbols_local.append(local_sym)
                            message_symbols_abs.append(abs_sym)
                        
                        # Get the input and output grids
                        input_np = input_tensor[0].detach().cpu().long().numpy().tolist()
                        output_np = output_tensor[0].detach().cpu().long().numpy().tolist()
                        
                        # Store selection sample
                        self.last_selection_sample = {
                            'direction': 'A1_to_A2',
                            'message_symbols_local': message_symbols_local,
                            'message_symbols_abs': message_symbols_abs,
                            'message_symbol_local': message_symbols_local[0] if message_symbols_local else 0,
                            'message_symbol_abs': message_symbols_abs[0] if message_symbols_abs else 0,
                            'sequence_length': seq_len,
                            'input_puzzle': input_np,
                            'output_puzzle': output_np,
                            'input_size': [input_tensor.shape[1], input_tensor.shape[2]],
                            'output_size': [output_tensor.shape[1], output_tensor.shape[2]],
                            'selection_correct': (pred1 == target_idx[0]).item(),
                            'confidence': correct_confidence1
                        }
                        
                        # Write to status file for web interface
                        try:
                            import os
                            status_path = globals().get('global_status_file_path', 'training_status.json')
                            if os.path.exists(status_path):
                                cur = {}
                                with open(status_path, 'r') as f:
                                    cur = json.load(f)
                                cur['last_selection_sample'] = self.last_selection_sample
                                with open(status_path, 'w') as f:
                                    json.dump(cur, f)
                        except Exception:
                            pass
            except Exception:
                pass
            
            metrics_history.append(metrics)
            
            # === NEW: If background decoder training is enabled, enqueue successful communications ===
            if self.decoder_background_enabled:
                # For Agent1→Agent2 direction
                # Enqueue (message, output_target) pairs for successful communications
                if int(pred1.item()) == 0:
                    try:
                        self.agent1.enqueue_successful_reconstruction(symbols1, output_tensor)
                    except Exception:
                        pass
                # For Agent2→Agent1 direction
                if int(pred2.item()) == 0:
                    try:
                        self.agent2.enqueue_successful_reconstruction(symbols2, output_tensor)
                    except Exception:
                        pass
            
        return metrics_history

    def set_reconstruction_mode(self, enabled: bool):
        """Enable or disable reconstruction-mode training at runtime."""
        enabled = bool(enabled)
        if enabled and not self.reconstruction_mode:
            print("[Trainer] Switching to RECONSTRUCTION mode")
            # Lazy-create decoder optimizer covering both agents' decoders
            if self._opt_decoders is None:
                self._opt_decoders = optim.Adam(
                    list(self.agent1.decoder.parameters()) + list(self.agent2.decoder.parameters()),
                    lr=self.learning_rate
                )
            self.reconstruction_mode = True
        elif (not enabled) and self.reconstruction_mode:
            print("[Trainer] Switching to SELECTION mode")
            self.reconstruction_mode = False

        # Reflect mode into training_mode string for metrics visibility
        if self.reconstruction_mode:
            self.training_mode = "reconstruction"
        else:
            # keep existing logic based on phase
            self.training_mode = self.training_mode if self.training_mode else "joint"

    def _train_reconstruction_step(self, puzzle: torch.Tensor, puzzle_idx: int, temperature: float = 1.0) -> Dict[str, float]:
        """
        Train both directions on reconstruction objective:
        Agent1 encodes INPUT → Agent2 decodes to reconstruct OUTPUT.
        Agent2 encodes INPUT → Agent1 decodes to reconstruct OUTPUT.
        Optimizes encoders, communication embeddings, message pooling, and both decoders.
        """
        # Ensure decoders optimizer exists
        if self._opt_decoders is None:
            self._opt_decoders = optim.Adam(
                list(self.agent1.decoder.parameters()) + list(self.agent2.decoder.parameters()),
                lr=self.learning_rate
            )

        # Ensure all necessary components have gradients enabled
        self.set_training_mode("joint")

        self.opt1.zero_grad()
        self.opt2.zero_grad()
        self._opt_decoders.zero_grad()

        # Get input and output tensors
        input_tensor = puzzle  # INPUT [1, H, W]
        puzzle_obj = self.active_puzzles[puzzle_idx]
        output_tensor = torch.tensor(
            puzzle_obj.test_output,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # OUTPUT [1, H, W]

        # Direction A1 -> A2: encode INPUT, reconstruct OUTPUT
        symbols1, _, _ = self.agent1.encode_puzzle_to_message(
            input_tensor, temperature=temperature, initial_phase=False
        )
        # Extract all symbols from message (for multi-position messages)
        with torch.no_grad():
            seq_len = symbols1.size(1) if symbols1.dim() == 3 else 1
            message_symbols_local = []
            message_symbols_abs = []
            
            for pos in range(seq_len):
                if symbols1.dim() == 3:
                    local_sym = int(symbols1[0, pos].argmax().item())
                else:
                    local_sym = 0
                abs_sym = int(self.agent1.puzzle_symbols + local_sym)
                message_symbols_local.append(local_sym)
                message_symbols_abs.append(abs_sym)
            
            # Keep backward compatibility variables
            local_sym1 = message_symbols_local[0] if message_symbols_local else 0
            abs_sym1 = message_symbols_abs[0] if message_symbols_abs else 0
        # Embed message with receiver's comm embeddings
        comm_emb_a2 = self.agent2.communication_embedding.weight[
            self.agent2.puzzle_symbols:self.agent2.current_total_symbols
        ]
        embedded_msg1 = torch.matmul(symbols1, comm_emb_a2)
        # Pass input_tensor to decoder so it can start from the input grid
        logits1, _, _, (hlog1, wlog1) = self.agent2.decoder(embedded_msg1, temperature=1.0, input_grid=input_tensor)
        B1, Hp1, Wp1, C1 = logits1.shape
        Ht, Wt = int(output_tensor.shape[1]), int(output_tensor.shape[2])
        Hc1, Wc1 = min(Hp1, Ht), min(Wp1, Wt)
        recon_loss1 = F.cross_entropy(
            logits1[:, :Hc1, :Wc1, :].reshape(B1 * Hc1 * Wc1, C1),
            output_tensor[:, :Hc1, :Wc1].reshape(B1 * Hc1 * Wc1)
        )
        # Reconstruction accuracy A1->A2
        with torch.no_grad():
            pred_grid1 = logits1[:, :Hc1, :Wc1, :].argmax(dim=-1)
            recon_acc1 = (pred_grid1 == output_tensor[:, :Hc1, :Wc1]).float().mean().item()
        h_tgt_idx1 = torch.tensor([max(1, min(Ht, getattr(self.agent2.decoder, 'max_height', Ht))) - 1], device=self.device)
        w_tgt_idx1 = torch.tensor([max(1, min(Wt, getattr(self.agent2.decoder, 'max_width', Wt))) - 1], device=self.device)
        size_loss1 = F.cross_entropy(hlog1, h_tgt_idx1) + F.cross_entropy(wlog1, w_tgt_idx1)

        # Direction A2 -> A1: encode INPUT, reconstruct OUTPUT
        symbols2, _, _ = self.agent2.encode_puzzle_to_message(
            input_tensor, temperature=temperature, initial_phase=False
        )
        with torch.no_grad():
            local_sym2 = int(symbols2[0, 0].argmax().item()) if symbols2.dim() == 3 else 0
            abs_sym2 = int(self.agent2.puzzle_symbols + local_sym2)
        comm_emb_a1 = self.agent1.communication_embedding.weight[
            self.agent1.puzzle_symbols:self.agent1.current_total_symbols
        ]
        embedded_msg2 = torch.matmul(symbols2, comm_emb_a1)
        # Pass input_tensor to decoder so it can start from the input grid
        logits2, _, _, (hlog2, wlog2) = self.agent1.decoder(embedded_msg2, temperature=1.0, input_grid=input_tensor)
        B2, Hp2, Wp2, C2 = logits2.shape
        Hc2, Wc2 = min(Hp2, Ht), min(Wp2, Wt)
        recon_loss2 = F.cross_entropy(
            logits2[:, :Hc2, :Wc2, :].reshape(B2 * Hc2 * Wc2, C2),
            output_tensor[:, :Hc2, :Wc2].reshape(B2 * Hc2 * Wc2)
        )
        # Reconstruction accuracy A2->A1
        with torch.no_grad():
            pred_grid2 = logits2[:, :Hc2, :Wc2, :].argmax(dim=-1)
            recon_acc2 = (pred_grid2 == output_tensor[:, :Hc2, :Wc2]).float().mean().item()
        h_tgt_idx2 = torch.tensor([max(1, min(Ht, getattr(self.agent1.decoder, 'max_height', Ht))) - 1], device=self.device)
        w_tgt_idx2 = torch.tensor([max(1, min(Wt, getattr(self.agent1.decoder, 'max_width', Wt))) - 1], device=self.device)
        size_loss2 = F.cross_entropy(hlog2, h_tgt_idx2) + F.cross_entropy(wlog2, w_tgt_idx2)

        total_loss = recon_loss1 + recon_loss2 + 0.1 * (size_loss1 + size_loss2)
        total_loss.backward()

        # Step optimizers
        self.opt1.step()
        self.opt2.step()
        self._opt_decoders.step()

        # Optional: include a lightweight reconstruction sample intermittently
        self._recon_step_counter += 1
        recon_sample = None
        try:
            if self._recon_step_counter % max(1, int(self.recon_sample_interval)) == 0:
                # FIXED: Show full grids at their actual sizes, not cropped
                # Display OUTPUT as target (what should be reconstructed)
                tgt_np = output_tensor[0].detach().cpu().long().numpy().tolist()
                pred_np = logits1[0].argmax(dim=-1).detach().cpu().long().numpy().tolist()
                recon_sample = {
                    'direction': 'A1_to_A2',
                    'message_symbols_local': message_symbols_local,
                    'message_symbols_abs': message_symbols_abs,
                    # Keep backward compatibility
                    'message_symbol_local': local_sym1,
                    'message_symbol_abs': abs_sym1,
                    'sequence_length': seq_len,
                    'target': tgt_np,
                    'reconstruction': pred_np,
                    'target_size': [Ht, Wt],
                    'predicted_size': [Hp1, Wp1]
                }
        except Exception:
            recon_sample = None

        metrics = {
            'cycle': self.cycle_count,
            'phase': self.current_phase,
            'phase_cycle': self.phase_cycle,
            'global_phase_count': self.global_phase_count,
            'total_loss': float(total_loss.item()),
            'recon_loss_a1_to_a2': float(recon_loss1.item()),
            'recon_loss_a2_to_a1': float(recon_loss2.item()),
            'reconstruction_acc_a1_to_a2': float(recon_acc1),
            'reconstruction_acc_a2_to_a1': float(recon_acc2),
            'reconstruction_accuracy': float((recon_acc1 + recon_acc2) / 2.0),
            'training_mode': 'reconstruction',
            'active_puzzles': len(self.active_puzzles),
            'mapped_puzzles': len(self.puzzle_symbol_mapping),
        }
        if recon_sample is not None:
            metrics['recon_sample'] = recon_sample
        return metrics
    
    def _calculate_symbol_entropy(self, symbol_probs):
        """Calculate the entropy of symbol probability distributions."""
        eps = 1e-8
        probs = symbol_probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean()

    # --- NEW: Unseen puzzle generalization testing ---
    def test_unseen_communication(self, num_tests: int = 100, temperature: float = 0.1, log_file_path: str = "unseen_testing_log.txt", bidirectional: bool = False) -> dict:
        """
        Evaluate sender→receiver communication on puzzles NOT used in training (unseen).
        Performs `num_tests` independent trials. For each trial, sample a target unseen puzzle
        and build a candidate set of 1 target + `num_distractors` distractors (prefer unseen; fallback to all).
        Writes detailed results for each trial to `log_file_path`.
        Returns a summary dict with counts and accuracy.
        """
        # Determine unseen dataset indices
        all_indices = set(range(len(self.available_arc_puzzles)))
        unseen_indices = list(all_indices - set(self.used_puzzle_indices))
        if len(all_indices) == 0:
            print("No available puzzles in dataset for unseen testing.")
            return {"num_tests": 0, "correct": 0, "accuracy": 0.0}
        if len(unseen_indices) == 0:
            print("Warning: No unseen puzzles remaining. Falling back to sampling from full dataset for testing.")
            unseen_indices = list(all_indices)
        
        # Use fixed distractor count for unseen test
        fixed_distractors = 3
        training_phase_distractors = getattr(self, 'last_training_phase_distractors', self.num_distractors)
        
        # Helper: choose targets (with replacement if needed)
        if len(unseen_indices) >= num_tests:
            target_dataset_indices = random.sample(unseen_indices, num_tests)
        else:
            target_dataset_indices = [random.choice(unseen_indices) for _ in range(num_tests)]
        
        # Prepare for eval
        self.agent1.eval()
        self.agent2.eval()
        correct = 0
        correct_rev = 0
        results = []
        results_rev = []
        
        # Open log file for appending
        try:
            log_f = open(log_file_path, 'a')
        except Exception:
            log_f = None
        
        # Header
        header = (
            f"\n{'='*60}\n"
            f"UNSEEN TESTING AFTER TRAINING PHASE\n"
            f"Global phase count: {self.global_phase_count}\n"
            f"Active puzzles: {len(self.active_puzzles)} | Unused in dataset: {len(all_indices - set(self.used_puzzle_indices))}\n"
            f"Num tests: {num_tests} | Distractors per test: {fixed_distractors}\n"
            f"Training phase distractors: {training_phase_distractors}\n"
            f"{'='*60}\n"
        )
        if log_f:
            log_f.write(header)
        else:
            print(header)
        # Log GES (MA) at test time
        ges1_ma_val, ges2_ma_val = self._current_ges_ma()
        if log_f:
            log_f.write(f"GES (MA) at test time: Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}\n")
        else:
            print(f"GES (MA) at test time: Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
        
        with torch.no_grad():
            for t_idx, dataset_idx in enumerate(target_dataset_indices, start=1):
                # Build target puzzle (input-output pair)
                target_puzzle = self.available_arc_puzzles[dataset_idx]
                # INPUT tensor for encoding
                input_tensor = torch.tensor(target_puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                # OUTPUT tensor for selection
                output_tensor = torch.tensor(target_puzzle.test_output, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Build distractors: prefer unseen (excluding target), then fallback to all
                candidate_distractors_pool = list(set(unseen_indices) - {dataset_idx})
                if len(candidate_distractors_pool) < fixed_distractors:
                    fallback_pool = list(all_indices - {dataset_idx})
                    candidate_distractors_pool = fallback_pool
                if len(candidate_distractors_pool) < fixed_distractors:
                    # If still not enough, allow repeats
                    distractor_dataset_indices = [random.choice(candidate_distractors_pool) for _ in range(fixed_distractors)]
                else:
                    distractor_dataset_indices = random.sample(candidate_distractors_pool, fixed_distractors)
                
                # Convert candidates to OUTPUT tensors (what receiver must select)
                candidates = [output_tensor]  # Target OUTPUT
                for d_idx in distractor_dataset_indices:
                    dp = self.available_arc_puzzles[d_idx]
                    # Use OUTPUT puzzle as candidate
                    cand_tensor = torch.tensor(dp.test_output, dtype=torch.long, device=self.device).unsqueeze(0)
                    candidates.append(cand_tensor)
                candidate_indices = [dataset_idx] + distractor_dataset_indices
                
                # Sender encodes INPUT → Receiver selects OUTPUT
                symbols, _, _ = self.agent1.encode_puzzle_to_message(
                    input_tensor, temperature=temperature, deterministic=True
                )
                selection_probs, selection_logits, _ = self.agent2.select_from_candidates(
                    symbols, candidates, temperature=temperature
                )
                predicted_idx = int(selection_logits.argmax(dim=-1).item())
                is_correct = (predicted_idx == 0)
                if is_correct:
                    correct += 1
                
                # Collect per-test details
                probs_list = [float(selection_probs[0, i].item()) for i in range(selection_probs.shape[1])]
                test_record = {
                    "test_number": t_idx,
                    "target_dataset_idx": dataset_idx,
                    "candidate_dataset_indices": candidate_indices,
                    "predicted_candidate": predicted_idx,
                    "correct": bool(is_correct),
                    "direction": "A1_to_A2",
                    "selection_probs": probs_list
                }
                results.append(test_record)
                
                # Log per-test details
                if log_f:
                    log_f.write(
                        f"Test {t_idx:03d} [A1→A2]: target={dataset_idx}, predicted={predicted_idx}, correct={'✓' if is_correct else '✗'}\n"
                    )
                    for i, (cand_idx, prob) in enumerate(zip(candidate_indices, probs_list)):
                        marker = ' (target)' if i == 0 else ''
                        log_f.write(f"  cand[{i}] dataset_idx={cand_idx} prob={prob:.4f}{marker}\n")

                # Optional reverse direction
                if bidirectional:
                    # Receiver becomes sender: encodes INPUT → Agent1 selects OUTPUT
                    symbols_rev, _, _ = self.agent2.encode_puzzle_to_message(
                        input_tensor, temperature=temperature, deterministic=True
                    )
                    selection_probs_rev, selection_logits_rev, _ = self.agent1.select_from_candidates(
                        symbols_rev, candidates, temperature=temperature
                    )
                    predicted_idx_rev = int(selection_logits_rev.argmax(dim=-1).item())
                    is_correct_rev = (predicted_idx_rev == 0)
                    if is_correct_rev:
                        correct_rev += 1
                    probs_list_rev = [float(selection_probs_rev[0, i].item()) for i in range(selection_probs_rev.shape[1])]
                    test_record_rev = {
                        "test_number": t_idx,
                        "target_dataset_idx": dataset_idx,
                        "candidate_dataset_indices": candidate_indices,
                        "predicted_candidate": predicted_idx_rev,
                        "correct": bool(is_correct_rev),
                        "direction": "A2_to_A1",
                        "selection_probs": probs_list_rev
                    }
                    results_rev.append(test_record_rev)
                    if log_f:
                        log_f.write(
                            f"Test {t_idx:03d} [A2→A1]: target={dataset_idx}, predicted={predicted_idx_rev}, correct={'✓' if is_correct_rev else '✗'}\n"
                        )
                        for i, (cand_idx, prob) in enumerate(zip(candidate_indices, probs_list_rev)):
                            marker = ' (target)' if i == 0 else ''
                            log_f.write(f"  cand[{i}] dataset_idx={cand_idx} prob={prob:.4f}{marker}\n")
        
        # Summary
        total_tests = num_tests * (2 if bidirectional else 1)
        total_correct = correct + (correct_rev if bidirectional else 0)
        accuracy = (total_correct / total_tests) if total_tests > 0 else 0.0
        acc_a1_a2 = (correct / num_tests) if num_tests > 0 else 0.0
        acc_a2_a1 = (correct_rev / num_tests) if (bidirectional and num_tests > 0) else None
        combined_results = results + (results_rev if bidirectional else [])
        # Reuse GES (MA) measured at test start
        ges1_ma_val, ges2_ma_val = self._current_ges_ma()
        summary = {
            "num_tests": total_tests,
            "correct": total_correct,
            "accuracy": accuracy,
            "a1_to_a2_correct": correct,
            "a2_to_a1_correct": (correct_rev if bidirectional else None),
            "a1_to_a2_accuracy": acc_a1_a2,
            "a2_to_a1_accuracy": acc_a2_a1,
            "ges1_ma": ges1_ma_val,
            "ges2_ma": ges2_ma_val,
            "results": combined_results,
            "distractors_per_test": fixed_distractors,
            "training_phase_distractors": training_phase_distractors
        }
        
        if log_f:
            log_f.write("Summary (bidirectional):\n")
            log_f.write(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})\n")
            if bidirectional:
                log_f.write(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})\n")
            log_f.write(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})\n")
            log_f.write(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}\n")
            log_f.write(f"  Training phase distractors: {training_phase_distractors}\n")
            log_f.flush()
            log_f.close()
        else:
            print("Summary (bidirectional):")
            print(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})")
            if bidirectional:
                print(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})")
            print(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})")
            print(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
            print(f"  Training phase distractors: {training_phase_distractors}")
        
        # Restore train mode
        self.agent1.train()
        self.agent2.train()
        
        return summary

    def _update_ges_moving_averages(self, metrics: Dict[str, float]):
        """
        NEW: Update internal GES moving averages given a step's metrics.
        """
        try:
            chance = 1.0 / max(1, metrics.get('num_candidates', self.num_distractors + 1))
            active_puzzles = max(1, metrics.get('active_puzzles', len(self.active_puzzles)))
            symbols = metrics.get('mapped_puzzles', len(self.puzzle_symbol_mapping))
            ratio = (active_puzzles / symbols) if symbols and symbols > 0 else float('nan')
            ges1_val = ((metrics['agent1_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else float('nan')
            ges2_val = ((metrics['agent2_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else float('nan')
        except Exception:
            ges1_val, ges2_val = float('nan'), float('nan')
        if not np.isnan(ges1_val):
            self._ges1_values.append(ges1_val)
            if len(self._ges1_values) > self._ges_window:
                self._ges1_values.pop(0)
        if not np.isnan(ges2_val):
            self._ges2_values.append(ges2_val)
            if len(self._ges2_values) > self._ges_window:
                self._ges2_values.pop(0)

    def _current_ges_ma(self) -> Tuple[float, float]:
        """
        NEW: Return current moving average GES for both agents.
        """
        ma1 = float(np.mean(self._ges1_values)) if len(self._ges1_values) > 0 else float('nan')
        ma2 = float(np.mean(self._ges2_values)) if len(self._ges2_values) > 0 else float('nan')
        return ma1, ma2

    def freeze_all(self):
        """
        NEW: Freeze all model parameters and disable optimizers' gradients.
        """
        self.agent1.freeze_all_parameters()
        self.agent2.freeze_all_parameters()
        # Put in eval mode for safety
        self.agent1.eval()
        self.agent2.eval()

    def unfreeze_all(self):
        """
        NEW: Un-freeze all model parameters and return to train mode.
        """
        for p in self.agent1.parameters():
            p.requires_grad = True
        for p in self.agent2.parameters():
            p.requires_grad = True
        self.agent1.train()
        self.agent2.train()

    def run_novel_symbol_induction_test(self, num_tests: int = 100, temperature: float = 0.1, log_file_path: str = "novel_symbol_unseen_testing_log.txt", bidirectional: bool = False, log_summary_only: bool = False, disable_file_logging: bool = False) -> dict:
        """
        NEW: After freezing, evaluate on unseen puzzles by inducing a novel symbol per puzzle.
        For each unseen test:
          - Encode target puzzle to sequence embedding
          - Predict a symbol embedding (position 0)
          - Allocate new symbol at that embedding on BOTH agents
          - Sender sends that single-symbol message; receiver performs selection among 1+num_distractors candidates
        Returns summary dict.
        """
        # Determine unseen dataset indices
        all_indices = set(range(len(self.available_arc_puzzles)))
        unseen_indices = list(all_indices - set(self.used_puzzle_indices))
        if len(all_indices) == 0:
            print("No available puzzles in dataset for novel symbol testing.")
            return {"num_tests": 0, "correct": 0, "accuracy": 0.0}
        if len(unseen_indices) == 0:
            print("Warning: No unseen puzzles remaining. Falling back to sampling from full dataset for testing.")
            unseen_indices = list(all_indices)
        
        # Use fixed distractor count for this test
        fixed_distractors = 3
        training_phase_distractors = getattr(self, 'last_training_phase_distractors', self.num_distractors)
        
        # Choose targets
        if len(unseen_indices) >= num_tests:
            target_dataset_indices = random.sample(unseen_indices, num_tests)
        else:
            target_dataset_indices = [random.choice(unseen_indices) for _ in range(num_tests)]
        
        # Ensure eval and frozen
        self.freeze_all()
        
        # Snapshot communication embedding tables and vocab sizes so test is non-destructive
        snapshot = {
            'a1_weight': self.agent1.communication_embedding.weight.detach().clone(),
            'a2_weight': self.agent2.communication_embedding.weight.detach().clone(),
            'a1_comm': self.agent1.current_comm_symbols,
            'a2_comm': self.agent2.current_comm_symbols,
            'a1_total': self.agent1.current_total_symbols,
            'a2_total': self.agent2.current_total_symbols,
            'a1_vocab': set(self.agent1.communication_vocabulary),
            'a2_vocab': set(self.agent2.communication_vocabulary),
        }

        correct = 0
        correct_rev = 0
        results = []
        results_rev = []
        # Open log file
        try:
            log_f = None if disable_file_logging else open(log_file_path, 'a')
        except Exception:
            log_f = None
        
        header = (
            f"\n{'='*60}\n"
            f"NOVEL SYMBOL INDUCTION TEST (FROZEN MODELS)\n"
            f"Num tests: {num_tests} | Distractors per test: {fixed_distractors}\n"
            f"Training phase distractors: {training_phase_distractors}\n"
            f"{'='*60}\n"
        )
        if log_f and not log_summary_only:
            log_f.write(header)
        elif not log_f:
            print(header)
        # Log GES (MA) at test time
        ges1_ma_val, ges2_ma_val = self._current_ges_ma()
        if log_f:
            log_f.write(f"GES (MA) at test time: Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}\n")
        else:
            print(f"GES (MA) at test time: Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
        
        with torch.no_grad():
            for t_idx, dataset_idx in enumerate(target_dataset_indices, start=1):
                # Build target puzzle (input-output pair)
                target_puzzle = self.available_arc_puzzles[dataset_idx]
                # INPUT for encoding
                input_tensor = torch.tensor(target_puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                # OUTPUT for selection
                output_tensor = torch.tensor(target_puzzle.test_output, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Prefer unseen for distractors if possible
                candidate_distractors_pool = list(set(unseen_indices) - {dataset_idx})
                if len(candidate_distractors_pool) < fixed_distractors:
                    fallback_pool = list(all_indices - {dataset_idx})
                    candidate_distractors_pool = fallback_pool
                if len(candidate_distractors_pool) < fixed_distractors:
                    distractor_dataset_indices = [random.choice(candidate_distractors_pool) for _ in range(fixed_distractors)]
                else:
                    distractor_dataset_indices = random.sample(candidate_distractors_pool, fixed_distractors)
                
                # Build OUTPUT candidates for selection
                candidates = [output_tensor]  # Target OUTPUT
                for d_idx in distractor_dataset_indices:
                    dp = self.available_arc_puzzles[d_idx]
                    # Use OUTPUT puzzle as candidate
                    cand_tensor = torch.tensor(dp.test_output, dtype=torch.long, device=self.device).unsqueeze(0)
                    candidates.append(cand_tensor)
                
                # 1) Predict embedding for INPUT puzzle using sender's encoder
                #    Create sequence embedding first
                seq_emb = self.agent1.embedding_system.embed_puzzle(input_tensor)  # [1, L, D]
                pred_emb = self.agent1.encoder.predict_symbol_embedding(seq_emb, position=0)  # [1, D]
                pred_emb = F.normalize(pred_emb, p=2, dim=-1)  # normalize for stability
                
                # 2) Allocate new symbol at this embedding on both agents
                new_sym_idx_1 = self.agent1.add_new_symbol_with_embedding(pred_emb)
                # Mirror onto agent2 to ensure consistent vocabulary
                self.agent2.add_new_symbol_with_embedding(pred_emb)
                
                # Build a one-hot-like message selecting the new symbol within current comm range
                # Translate absolute symbol to local comm index slice [puzzle_symbols:current_total)
                local_comm_index = new_sym_idx_1 - self.agent1.puzzle_symbols
                num_comm = self.agent1.current_comm_symbols
                seq_len = max(1, self.agent1.current_seq_length)
                message = torch.zeros((1, seq_len, num_comm), device=self.device)
                # Place the symbol at position 0
                if 0 <= local_comm_index < num_comm:
                    message[0, 0, local_comm_index] = 1.0
                else:
                    # Fallback: pick last available index
                    message[0, 0, num_comm - 1] = 1.0
                
                # 3) Receiver selects among candidates
                selection_probs, selection_logits, _ = self.agent2.select_from_candidates(
                    message, candidates, temperature=0.1
                )
                predicted_idx = int(selection_logits.argmax(dim=-1).item())
                is_correct = (predicted_idx == 0)
                if is_correct:
                    correct += 1
                
                # Log per-test
                probs_list = [float(selection_probs[0, i].item()) for i in range(selection_probs.shape[1])]
                candidate_indices = [dataset_idx] + distractor_dataset_indices
                if log_f and not log_summary_only:
                    log_f.write(
                        f"Test {t_idx:03d}: target={dataset_idx}, predicted={predicted_idx}, correct={'✓' if is_correct else '✗'}\n"
                    )
                    for i, (cand_idx, prob) in enumerate(zip(candidate_indices, probs_list)):
                        marker = ' (target)' if i == 0 else ''
                        log_f.write(f"  cand[{i}] dataset_idx={cand_idx} prob={prob:.4f}{marker}\n")
                
                results.append({
                    'test_number': t_idx,
                    'target_dataset_idx': dataset_idx,
                    'predicted_candidate': predicted_idx,
                    'correct': bool(is_correct),
                    'selection_probs': probs_list,
                    'candidate_dataset_indices': candidate_indices,
                    'direction': 'A1_to_A2'
                })

                # Optional reverse direction
                if bidirectional:
                    # Predict embedding using receiver's encoder (now acting as sender) on INPUT
                    seq_emb_rev = self.agent2.embedding_system.embed_puzzle(input_tensor)
                    pred_emb_rev = self.agent2.encoder.predict_symbol_embedding(seq_emb_rev, position=0)
                    pred_emb_rev = F.normalize(pred_emb_rev, p=2, dim=-1)
                    new_sym_idx_2 = self.agent2.add_new_symbol_with_embedding(pred_emb_rev)
                    self.agent1.add_new_symbol_with_embedding(pred_emb_rev)

                    local_comm_index_rev = new_sym_idx_2 - self.agent2.puzzle_symbols
                    num_comm_rev = self.agent2.current_comm_symbols
                    seq_len_rev = max(1, self.agent2.current_seq_length)
                    message_rev = torch.zeros((1, seq_len_rev, num_comm_rev), device=self.device)
                    if 0 <= local_comm_index_rev < num_comm_rev:
                        message_rev[0, 0, local_comm_index_rev] = 1.0
                    else:
                        message_rev[0, 0, num_comm_rev - 1] = 1.0

                    selection_probs_rev, selection_logits_rev, _ = self.agent1.select_from_candidates(
                        message_rev, candidates, temperature=0.1
                    )
                    predicted_idx_rev = int(selection_logits_rev.argmax(dim=-1).item())
                    is_correct_rev = (predicted_idx_rev == 0)
                    if is_correct_rev:
                        correct_rev += 1
                    probs_list_rev = [float(selection_probs_rev[0, i].item()) for i in range(selection_probs_rev.shape[1])]
                    if log_f and not log_summary_only:
                        log_f.write(
                            f"Test {t_idx:03d} [A2→A1]: target={dataset_idx}, predicted={predicted_idx_rev}, correct={'✓' if is_correct_rev else '✗'}\n"
                        )
                        for i, (cand_idx, prob) in enumerate(zip(candidate_indices, probs_list_rev)):
                            marker = ' (target)' if i == 0 else ''
                            log_f.write(f"  cand[{i}] dataset_idx={cand_idx} prob={prob:.4f}{marker}\n")
                    results_rev.append({
                        'test_number': t_idx,
                        'target_dataset_idx': dataset_idx,
                        'predicted_candidate': predicted_idx_rev,
                        'correct': bool(is_correct_rev),
                        'selection_probs': probs_list_rev,
                        'candidate_dataset_indices': candidate_indices,
                        'direction': 'A2_to_A1'
                    })
        
        total_tests = num_tests * (2 if bidirectional else 1)
        total_correct = correct + (correct_rev if bidirectional else 0)
        accuracy = (total_correct / total_tests) if total_tests > 0 else 0.0
        acc_a1_a2 = (correct / num_tests) if num_tests > 0 else 0.0
        acc_a2_a1 = (correct_rev / num_tests) if (bidirectional and num_tests > 0) else None
        ges1_ma_val, ges2_ma_val = self._current_ges_ma()
        summary = {"num_tests": total_tests, "correct": total_correct, "accuracy": accuracy, "a1_to_a2_correct": correct, "a2_to_a1_correct": (correct_rev if bidirectional else None), "a1_to_a2_accuracy": acc_a1_a2, "a2_to_a1_accuracy": acc_a2_a1, "ges1_ma": ges1_ma_val, "ges2_ma": ges2_ma_val, "results": results + (results_rev if bidirectional else [])}
        if log_f:
            # Always write a concise summary when logging is enabled
            log_f.write("Summary (bidirectional):\n")
            log_f.write(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})\n")
            if bidirectional:
                log_f.write(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})\n")
            log_f.write(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})\n")
            log_f.write(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}\n")
            log_f.write(f"  Training phase distractors: {training_phase_distractors}\n")
            log_f.flush()
            log_f.close()
        else:
            print("Summary (bidirectional):")
            print(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})")
            if bidirectional:
                print(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})")
            print(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})")
            print(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
            print(f"  Training phase distractors: {training_phase_distractors}")
        
        # Restore embedding tables and vocab sizes (non-destructive test)
        with torch.no_grad():
            self.agent1.communication_embedding.weight.copy_(snapshot['a1_weight'])
            self.agent2.communication_embedding.weight.copy_(snapshot['a2_weight'])
        self.agent1.current_comm_symbols = snapshot['a1_comm']
        self.agent2.current_comm_symbols = snapshot['a2_comm']
        self.agent1.current_total_symbols = snapshot['a1_total']
        self.agent2.current_total_symbols = snapshot['a2_total']
        self.agent1.communication_vocabulary = snapshot['a1_vocab']
        self.agent2.communication_vocabulary = snapshot['a2_vocab']

        # Unfreeze for subsequent training
        self.unfreeze_all()
        
        return summary

    def get_phase_status(self) -> Dict[str, any]:
        """
        Enhanced to include random selection tracking information.
        """
        base_status = {
            'current_phase': self.current_phase,
            'phase_cycle': self.phase_cycle,
            'global_phase_count': self.global_phase_count,
            'active_puzzles': len(self.active_puzzles),
            'removed_symbols': len(self.removed_symbols),
            'agent1_vocab': self.agent1.get_vocabulary_info(),
            'agent2_vocab': self.agent2.get_vocabulary_info(),
            'puzzle_symbol_mapping': self.puzzle_symbol_mapping,
            # NEW: Random selection tracking
            'used_puzzle_indices': len(self.used_puzzle_indices),
            'total_available_puzzles': len(self.available_arc_puzzles),
            'remaining_unused_puzzles': len(self.available_arc_puzzles) - len(self.used_puzzle_indices),
            'selection_strategy': 'random',  # Indicate this is using random selection
            'selection_config': {
                'num_distractors': self.num_distractors,
                'distractor_strategy': self.distractor_strategy,
                'first_training_cycles': self.first_training_cycles,
                'training_cycles': self.training_cycles,
                'consolidation_tests': self.consolidation_tests,
                'puzzles_per_addition': self.puzzles_per_addition,
                'repetitions_per_puzzle': self.repetitions_per_puzzle,
                'initial_puzzle_count': self.initial_puzzle_count,
                'initial_comm_symbols': self.initial_comm_symbols
            }
        }
        return base_status

    def save_snapshot(self, name: Optional[str] = None, directory: Optional[str] = None) -> str:
        """Save a full snapshot of both agents and relevant trainer state.
        
        For progressive sequence training:
        - Saves current communication symbols (will be frozen in next phase)
        - Saves current frozen positions and symbols
        
        Returns the saved filepath.
        """
        # Determine directory
        out_dir = directory or './outputs'
        snap_dir = os.path.join(out_dir, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)

        # Build filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = name.strip().replace(' ', '_') if name else f'comm_snapshot'
        filename = f"{base}_{timestamp}.pt"
        path = os.path.join(snap_dir, filename)
        
        # Update frozen_comm_symbols based on currently frozen count in agents
        # (in case it wasn't set explicitly during snapshot)
        if self.frozen_comm_symbols == 0 and hasattr(self.agent1, 'get_frozen_communication_symbols'):
            self.frozen_comm_symbols = self.agent1.get_frozen_communication_symbols()

        # Collect states
        state = {
            'agent1_state_dict': self.agent1.state_dict(),
            'agent2_state_dict': self.agent2.state_dict(),
            'architecture': {
                'embedding_dim': self.agent1.embedding_system.embedding_dim,
                'hidden_dim': getattr(self.agent1.encoder, 'hidden_dim', 1024),
                'num_symbols': self.agent1.max_num_symbols,
                'puzzle_symbols': self.agent1.puzzle_symbols,
                'max_seq_length': self.agent1.max_seq_length,
                'similarity_metric': self.agent1.similarity_metric,
            },
            'trainer_state': {
                'current_phase': self.current_phase,
                'phase_cycle': self.phase_cycle,
                'global_phase_count': self.global_phase_count,
                'cycle_count': self.cycle_count,
                'num_distractors': self.num_distractors,
                'distractor_strategy': self.distractor_strategy,
                'puzzle_symbol_mapping': self.puzzle_symbol_mapping,
                'symbol_puzzle_mapping': self.symbol_puzzle_mapping,
                'used_puzzle_indices': list(self.used_puzzle_indices),
                'current_comm_symbols_a1': getattr(self.agent1, 'current_comm_symbols', None),
                'current_comm_symbols_a2': getattr(self.agent2, 'current_comm_symbols', None),
                'current_total_symbols_a1': getattr(self.agent1, 'current_total_symbols', None),
                'current_total_symbols_a2': getattr(self.agent2, 'current_total_symbols', None),
                'current_seq_length': self.agent1.current_seq_length,
                'frozen_positions': getattr(self, 'frozen_positions', []),
                'frozen_comm_symbols': getattr(self, 'frozen_comm_symbols', 0),
                'position_vocabularies_a1': {int(k): list(v) for k, v in self.agent1.get_position_vocabularies().items()},
                'position_vocabularies_a2': {int(k): list(v) for k, v in self.agent2.get_position_vocabularies().items()},
                'initial_puzzle_count': self.initial_puzzle_count,
                'initial_comm_symbols': self.initial_comm_symbols,
                'repetition_per_puzzle': self.repetitions_per_puzzle,
                'learning_rate': self.learning_rate,
                'web_mode': self.web_mode,
            },
            'meta': {
                'created_at': timestamp,
                'name': name or '',
                'format_version': 1,
                'kind': 'communication_system_snapshot'
            }
        }

        torch.save(state, path)
        # Retention: keep only the most recent N snapshots
        try:
            max_keep = int(getattr(self, 'max_snapshots', 50) or 50)
            files = [f for f in os.listdir(snap_dir) if f.endswith('.pt')]
            files_full = [os.path.join(snap_dir, f) for f in files]
            files_full.sort(key=os.path.getmtime, reverse=True)
            for old in files_full[max_keep:]:
                try:
                    os.remove(old)
                except Exception:
                    pass
        except Exception:
            pass
        return path

    # NEW: mark and update helpers for distractor scheduling
    def _mark_ges_threshold_hit(self):
        if self.ges_threshold_hit_phase is None:
            self.ges_threshold_hit_phase = self.global_phase_count
            self.base_num_distractors_at_threshold = self.num_distractors
            if not self.web_mode:
                print(f"[Distractors] GES threshold hit at global phase {self.ges_threshold_hit_phase}. Baseline distractors: {self.base_num_distractors_at_threshold}")
            # === NEW: Enable background decoder training when threshold is first hit ===
            if not self.decoder_background_enabled:
                self.agent1.enable_background_decoder_training()
                self.agent2.enable_background_decoder_training()
                self.decoder_background_enabled = True

    def _update_distractors_for_current_phase(self):
        if self.ges_threshold_hit_phase is None:
            return
        phase_diff = self.global_phase_count - self.ges_threshold_hit_phase
        if phase_diff < 10:
            return
        increments = int((phase_diff - 10) // 3) + 1
        base = self.base_num_distractors_at_threshold if self.base_num_distractors_at_threshold is not None else self.num_distractors
        desired = min(10, base + increments)
        desired = max(0, desired)
        if desired > self.num_distractors:
            old = self.num_distractors
            self.num_distractors = desired
            if not self.web_mode:
                print(f"[Distractors] Increasing distractors: {old} → {self.num_distractors} (phase {self.global_phase_count})")


# Create a factory function to replace the original CommunicationTrainer
def CommunicationTrainer(*args, **kwargs):
    """Factory function to create ProgressiveSelectionTrainer instances"""
    return ProgressiveSelectionTrainer(*args, **kwargs)