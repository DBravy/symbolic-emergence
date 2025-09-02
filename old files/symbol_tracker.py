import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

class SymbolTracker:
    """
    Tracks and analyzes symbol usage patterns of encoders during training.
    """
    def __init__(self, num_puzzles=2, num_symbols=2):
        self.num_puzzles = num_puzzles
        self.num_symbols = num_symbols
        
        # Storage for symbol logits per puzzle
        self.puzzle_to_symbol_history = {
            i: [] for i in range(num_puzzles)
        }
        
        # Track mutual information over time
        self.mutual_info_history = []
        
    def update(self, puzzle_idx: int, symbol_logits: torch.Tensor):
        """
        Update tracker with symbol logits for a specific puzzle.
        
        Args:
            puzzle_idx: Index of the puzzle
            symbol_logits: Raw logits from encoder before sampling [batch, seq, num_symbols]
        """
        # Convert to probabilities
        with torch.no_grad():
            symbol_probs = torch.softmax(symbol_logits, dim=-1).cpu().numpy()
            # Just take first item in batch and sequence
            if symbol_probs.shape[0] > 0 and symbol_probs.shape[1] > 0:
                self.puzzle_to_symbol_history[puzzle_idx].append(symbol_probs[0, 0])
                
                # Compute mutual information if we have data for all puzzles
                self._compute_mutual_information()
    
    def _compute_mutual_information(self):
        """
        Compute mutual information between puzzles and symbols.
        High MI indicates the encoder is using different symbols for different puzzles.
        """
        # Only compute if we have data for all puzzles
        if not all(len(history) > 0 for history in self.puzzle_to_symbol_history.values()):
            return
        
        # Get latest probabilities for each puzzle
        latest_probs = [history[-1] for history in self.puzzle_to_symbol_history.values()]
        
        # Compute joint distribution (puzzle, symbol)
        joint_dist = np.array(latest_probs)
        
        # Compute marginals
        puzzle_marginal = np.ones(self.num_puzzles) / self.num_puzzles  # Assuming uniform puzzle distribution
        symbol_marginal = joint_dist.mean(axis=0)
        
        # Compute MI
        mi = 0
        for i in range(self.num_puzzles):
            for j in range(self.num_symbols):
                p_joint = joint_dist[i, j]
                p_puzzle = puzzle_marginal[i]
                p_symbol = symbol_marginal[j]
                if p_joint > 0:
                    mi += p_joint * np.log2(p_joint / (p_puzzle * p_symbol))
        
        self.mutual_info_history.append(mi)
    
    def get_symbol_preference_strengths(self) -> Dict[int, float]:
        """
        Calculate how strongly each puzzle prefers one symbol over others.
        Returns a dictionary mapping puzzle indices to preference strengths.
        """
        strengths = {}
        for puzzle_idx, history in self.puzzle_to_symbol_history.items():
            if history:
                # Get latest probabilities
                latest_probs = history[-1]
                
                # Calculate entropy (lower means stronger preference)
                entropy = -np.sum(latest_probs * np.log2(latest_probs + 1e-10))
                max_entropy = np.log2(self.num_symbols)
                
                # Convert to a strength measure (0 to 1, higher is stronger preference)
                strength = 1.0 - (entropy / max_entropy)
                strengths[puzzle_idx] = strength
        
        return strengths
    
    def get_symbol_assignments(self) -> Dict[int, int]:
        """
        Determine which symbol each puzzle is most strongly assigned to.
        Returns a dictionary mapping puzzle indices to symbol indices.
        """
        assignments = {}
        for puzzle_idx, history in self.puzzle_to_symbol_history.items():
            if history:
                # Get latest probabilities
                latest_probs = history[-1]
                
                # Get the symbol with highest probability
                most_likely_symbol = np.argmax(latest_probs)
                assignments[puzzle_idx] = most_likely_symbol
        
        return assignments
    
    def plot_symbol_probabilities(self, output_dir='.'):
        """
        Plot symbol probability evolution for each puzzle.
        """
        for puzzle_idx, history in self.puzzle_to_symbol_history.items():
            if not history:
                continue
                
            plt.figure(figsize=(10, 5))
            
            # Convert list of probability arrays to array of shape [time_steps, num_symbols]
            probs_over_time = np.array(history)
            time_steps = np.arange(len(history))
            
            for symbol_idx in range(self.num_symbols):
                plt.plot(time_steps, probs_over_time[:, symbol_idx], 
                         label=f'Symbol {symbol_idx}')
            
            plt.title(f'Symbol Probabilities for Puzzle {puzzle_idx}')
            plt.xlabel('Training Step')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()
            
            # Save figure
            filename = os.path.join(output_dir, f'puzzle_{puzzle_idx}_symbol_probs.png')
            plt.savefig(filename)
            plt.close()
    
    def plot_mutual_information(self, output_dir='.'):
        """
        Plot mutual information over time.
        """
        if not self.mutual_info_history:
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.mutual_info_history)
        plt.title('Mutual Information between Puzzles and Symbols')
        plt.xlabel('Training Step')
        plt.ylabel('Mutual Information (bits)')
        plt.grid(True)
        
        # Theoretical maximum for 2 puzzles and 2 symbols is 1 bit
        if self.num_puzzles == 2 and self.num_symbols == 2:
            plt.axhline(y=1.0, color='r', linestyle='--', 
                       label='Theoretical Maximum (1 bit)')
            plt.legend()
        
        # Save figure
        filename = os.path.join(output_dir, 'mutual_information.png')
        plt.savefig(filename)
        plt.close()
    
    def get_summary(self) -> str:
        """
        Generate a human-readable summary of the current state.
        """
        summary = []
        summary.append("\nSymbol Usage Analysis:")
        
        # Current mutual information
        if self.mutual_info_history:
            current_mi = self.mutual_info_history[-1]
            max_mi = 1.0 if self.num_puzzles == 2 and self.num_symbols == 2 else "unknown"
            summary.append(f"  Mutual Information: {current_mi:.4f} bits (max: {max_mi})")
        
        # Symbol preference strengths
        strengths = self.get_symbol_preference_strengths()
        if strengths:
            summary.append("\n  Symbol Preference Strengths:")
            for puzzle_idx, strength in strengths.items():
                summary.append(f"    Puzzle {puzzle_idx}: {strength:.4f}")
        
        # Current symbol assignments
        assignments = self.get_symbol_assignments()
        if assignments:
            summary.append("\n  Current Symbol Assignments:")
            for puzzle_idx, symbol_idx in assignments.items():
                summary.append(f"    Puzzle {puzzle_idx} → Symbol {symbol_idx}")
        
        # Check if assignments are optimal
        if len(assignments) == self.num_puzzles:
            unique_symbols = len(set(assignments.values()))
            if unique_symbols == self.num_puzzles:
                summary.append("\n  Status: OPTIMAL - Each puzzle has a unique symbol!")
            else:
                summary.append("\n  Status: SUBOPTIMAL - Some puzzles share the same symbol")
        
        return "\n".join(summary)