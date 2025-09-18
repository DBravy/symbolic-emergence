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
        consolidation_tests: int = 10,         # Number of test cycles in consolidation
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
        self.initial_num_distractors = num_distractors
        self.distractor_strategy = distractor_strategy

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
        
        print(f"Initialized with {len(self.active_puzzles)} RANDOMLY SELECTED puzzles")
        print(f"Selected puzzle indices: {selected_indices}")
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
                puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)

                symbol = self.puzzle_symbol_mapping[puzzle_idx]
                if not self.web_mode:
                    print(f"\nTesting Puzzle {puzzle_idx} (Symbol {symbol}) - 10 selection tests:")

                correct = 0
                for test_num in range(10):
                    # Sender encodes → message
                    symbols, _, _ = self.agent1.encode_puzzle_to_message(
                        puzzle_tensor, temperature=0.1, deterministic=True
                    )

                    # Build candidates: target first, then distractors
                    distractors = self.sample_distractors(puzzle_tensor, puzzle_idx)
                    candidates = [puzzle_tensor] + distractors  # each [1, H, W]

                    # Receiver selects
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
        Modified to remove symbols with ≤30% accuracy (instead of just 0% accuracy).
        Shows visual debugging for poor performers, with robust fallbacks.
        Also writes all output to consolidation_analysis.txt.
        """
        recessive_symbols: Set[int] = set()
        poor_performers: List[int] = []  # Symbols with ≤30% accuracy (but >0%)

        consolidation_filename = 'consolidation_analysis.txt'

        if not self.web_mode:
            header = f"{'='*50}\nANALYZING SYMBOL PERFORMANCE (10 tests per symbol)\n{'='*50}"
            print(header)
        else:
            header = f"Analyzing symbol performance..."
            print(header)
        
        with open(consolidation_filename, 'a') as log_file:
            log_file.write(f"\n{header}\n")

        # --- Compute per-symbol accuracy and categorize ---
        for symbol, predictions in confusion_data.items():
            total_tests = len(predictions)
            correct_predictions = sum(1 for pred in predictions if pred == symbol)
            accuracy = (correct_predictions / total_tests) if total_tests > 0 else 0.0

            symbol_line = f"Symbol {symbol}: {correct_predictions}/{total_tests} tests correct ({accuracy:.1%})"
            if not self.web_mode:
                print(symbol_line)
            with open(consolidation_filename, 'a') as log_file:
                log_file.write(f"{symbol_line}\n")

            # CHANGED: Now include all symbols with ≤30% accuracy for removal
            if accuracy <= 0.30:
                recessive_symbols.add(symbol)
                if accuracy == 0.0:
                    status_line = f"  → RECESSIVE: Symbol {symbol} never selected correctly in any test"
                else:
                    status_line = f"  → RECESSIVE: Symbol {symbol} selected correctly ≤30% of the time"
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
        """Evaluate selection accuracy (Agent1 encodes → Agent2 selects) for given puzzles.
        Returns a dict of puzzle_idx -> number of correct selections out of `tests`.
        """
        self.agent1.eval()
        self.agent2.eval()
        results: Dict[int, int] = {}
        indices = puzzle_indices if puzzle_indices is not None else list(range(len(self.active_puzzles)))
        with torch.no_grad():
            for puzzle_idx in indices:
                puzzle = self.active_puzzles[puzzle_idx]
                puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                correct = 0
                for _ in range(tests):
                    symbols, _, _ = self.agent1.encode_puzzle_to_message(
                        puzzle_tensor, temperature=temperature, deterministic=True
                    )
                    distractors = self.sample_distractors(puzzle_tensor, puzzle_idx)
                    candidates = [puzzle_tensor] + distractors
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
            candidates1 = [puzzle] + distractors  # each [1, H, W]
            
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
        fixed_distractors = getattr(self, 'initial_num_distractors', self.num_distractors)
        
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
                # Build target tensor
                target_puzzle = self.available_arc_puzzles[dataset_idx]
                target_tensor = torch.tensor(target_puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                
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
                
                # Convert candidates to tensors
                candidates = [target_tensor]
                for d_idx in distractor_dataset_indices:
                    dp = self.available_arc_puzzles[d_idx]
                    cand_tensor = torch.tensor(dp.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                    candidates.append(cand_tensor)
                candidate_indices = [dataset_idx] + distractor_dataset_indices
                
                # Sender encodes → Receiver selects
                symbols, _, _ = self.agent1.encode_puzzle_to_message(
                    target_tensor, temperature=temperature, deterministic=True
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
                    # Receiver becomes sender → Agent1 selects
                    symbols_rev, _, _ = self.agent2.encode_puzzle_to_message(
                        target_tensor, temperature=temperature, deterministic=True
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
            "distractors_per_test": fixed_distractors
        }
        
        if log_f:
            log_f.write("Summary (bidirectional):\n")
            log_f.write(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})\n")
            if bidirectional:
                log_f.write(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})\n")
            log_f.write(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})\n")
            log_f.write(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}\n")
            log_f.flush()
            log_f.close()
        else:
            print("Summary (bidirectional):")
            print(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})")
            if bidirectional:
                print(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})")
            print(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})")
            print(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
        
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
        fixed_distractors = getattr(self, 'initial_num_distractors', self.num_distractors)
        
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
                # Build target and candidates
                target_puzzle = self.available_arc_puzzles[dataset_idx]
                target_tensor = torch.tensor(target_puzzle.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                # Prefer unseen for distractors if possible
                candidate_distractors_pool = list(set(unseen_indices) - {dataset_idx})
                if len(candidate_distractors_pool) < fixed_distractors:
                    fallback_pool = list(all_indices - {dataset_idx})
                    candidate_distractors_pool = fallback_pool
                if len(candidate_distractors_pool) < fixed_distractors:
                    distractor_dataset_indices = [random.choice(candidate_distractors_pool) for _ in range(fixed_distractors)]
                else:
                    distractor_dataset_indices = random.sample(candidate_distractors_pool, fixed_distractors)
                candidates = [target_tensor]
                for d_idx in distractor_dataset_indices:
                    dp = self.available_arc_puzzles[d_idx]
                    cand_tensor = torch.tensor(dp.test_input, dtype=torch.long, device=self.device).unsqueeze(0)
                    candidates.append(cand_tensor)
                
                # 1) Predict embedding for this new puzzle using sender's encoder
                #    Create sequence embedding first
                seq_emb = self.agent1.embedding_system.embed_puzzle(target_tensor)  # [1, L, D]
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
                    # Predict embedding using receiver's encoder (now acting as sender)
                    seq_emb_rev = self.agent2.embedding_system.embed_puzzle(target_tensor)
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
            log_f.flush()
            log_f.close()
        else:
            print("Summary (bidirectional):")
            print(f"  A1→A2: {correct}/{num_tests} correct (acc={acc_a1_a2:.3f})")
            if bidirectional:
                print(f"  A2→A1: {correct_rev}/{num_tests} correct (acc={acc_a2_a1:.3f})")
            print(f"  Overall: {total_correct}/{total_tests} correct (acc={accuracy:.3f})")
            print(f"  GES (MA): Agent1={ges1_ma_val:.2f}, Agent2={ges2_ma_val:.2f}")
        
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

    # NEW: mark and update helpers for distractor scheduling
    def _mark_ges_threshold_hit(self):
        if self.ges_threshold_hit_phase is None:
            self.ges_threshold_hit_phase = self.global_phase_count
            self.base_num_distractors_at_threshold = self.num_distractors
            if not self.web_mode:
                print(f"[Distractors] GES threshold hit at global phase {self.ges_threshold_hit_phase}. Baseline distractors: {self.base_num_distractors_at_threshold}")

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