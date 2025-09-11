import torch
# SELECTION MODIFICATION: Use selection versions
from agent_selection import ProgressiveSelectionAgent as Agent
from trainer_selection import ProgressiveSelectionTrainer as CommunicationTrainer

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.raise_window'] = False
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from puzzle import Puzzle
import numpy as np
import json
import os
import torch.nn.functional as F
from collections import deque
import random
import threading
import time

# Enable interactive mode for live plotting
plt.ion()

class LiveGrapher:
    def __init__(self, max_points=None, ma_window=50, initial_symbols=3):
        # max_points is kept for compatibility but not used (no sliding behavior)
        self.max_points = max_points
        self.ma_window = ma_window
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Live Training Progress', fontsize=16)
        
        # Track active symbols count with simple addition/subtraction
        self.active_symbols_count = initial_symbols
        
        # Data storage
        self.data = {
            'loss': [],
            'acc1_selection': [],
            'acc2_selection': [],
            'ges1': [],
            'ges2': [],
            'active_symbols': [],
            'steps': [],
            'total_puzzles': []
        }
        
        # Moving averages for smoothed display
        self.moving_averages = {
            'acc1_selection': MovingAverage(ma_window),
            'acc2_selection': MovingAverage(ma_window),
            'ges1': MovingAverage(ma_window),
            'ges2': MovingAverage(ma_window),
            'loss': MovingAverage(ma_window)
        }
        
        # Store moving average values for plotting
        self.ma_data = {
            'loss': [],
            'acc1_selection': [],
            'acc2_selection': [],
            'ges1': [],
            'ges2': []
        }
        
        # Phase tracking
        self.phase_markers = []
        self.phase_colors = {
            'pretraining': 'blue',
            'training': 'green', 
            'consolidation': 'orange',
            'addition': 'red',
            'remedial': 'magenta'
        }
        
        # Initialize plots
        self.setup_plots()
        
        # Thread safety
        self.lock = threading.Lock()
        
    def setup_plots(self):
        """Initialize all subplot configurations"""
        # Loss plot
        self.axes[0, 0].set_title(f'Training Loss (Log Scale, MA-{self.ma_window})')
        self.axes[0, 0].set_xlabel('Step')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].set_yscale('log')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Selection accuracies
        self.axes[0, 1].set_title(f'Selection Accuracies (MA-{self.ma_window})')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].set_ylim(0, 1.1)
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # GES plot (replaces confidence)
        self.axes[0, 2].set_title(f'Generalization Efficiency Score (GES) (MA-{self.ma_window})')
        self.axes[0, 2].set_xlabel('Step')
        self.axes[0, 2].set_ylabel('GES')
        # allow negative values; no fixed ylim
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Active symbols
        self.axes[1, 0].set_title('Active Symbols')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Count')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Phase timeline
        self.axes[1, 1].set_title('Training Phases')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Phase')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Statistics summary
        self.axes[1, 2].set_title('Current Statistics')
        self.axes[1, 2].axis('off')
        
        plt.tight_layout()
        
    def add_data_point(self, step, loss=None, acc1=None, acc2=None, 
                     ges1=None, ges2=None, phase=None, total_puzzles=None):
        """Add a new data point and update the graph"""
        with self.lock:
            self.data['steps'].append(step)
            self.data['loss'].append(loss if loss is not None else np.nan)
            self.data['acc1_selection'].append(acc1 if acc1 is not None else np.nan)
            self.data['acc2_selection'].append(acc2 if acc2 is not None else np.nan)
            self.data['ges1'].append(ges1 if ges1 is not None else np.nan)
            self.data['ges2'].append(ges2 if ges2 is not None else np.nan)
            self.data['active_symbols'].append(self.active_symbols_count)
            self.data['total_puzzles'].append(total_puzzles if total_puzzles is not None else np.nan)
            
            # Update moving averages and store MA values
            if loss is not None and not np.isnan(loss):
                self.moving_averages['loss'].update(loss)
                self.ma_data['loss'].append(self.moving_averages['loss'].get_average())
            else:
                self.ma_data['loss'].append(np.nan)
            
            if acc1 is not None and not np.isnan(acc1):
                self.moving_averages['acc1_selection'].update(acc1)
                self.ma_data['acc1_selection'].append(self.moving_averages['acc1_selection'].get_average())
            else:
                self.ma_data['acc1_selection'].append(np.nan)
            
            if acc2 is not None and not np.isnan(acc2):
                self.moving_averages['acc2_selection'].update(acc2)
                self.ma_data['acc2_selection'].append(self.moving_averages['acc2_selection'].get_average())
            else:
                self.ma_data['acc2_selection'].append(np.nan)
            
            if ges1 is not None and not np.isnan(ges1):
                self.moving_averages['ges1'].update(ges1)
                self.ma_data['ges1'].append(self.moving_averages['ges1'].get_average())
            else:
                self.ma_data['ges1'].append(np.nan)
            
            if ges2 is not None and not np.isnan(ges2):
                self.moving_averages['ges2'].update(ges2)
                self.ma_data['ges2'].append(self.moving_averages['ges2'].get_average())
            else:
                self.ma_data['ges2'].append(np.nan)
            
            # Track phase changes
            if phase:
                self.phase_markers.append((step, phase))
            
            # Keep all data points - no sliding behavior
            # Data will automatically scale to fit within the plot boundaries
            
            self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data"""
        try:
            steps = self.data['steps']
            if not steps:
                return
            
            # Clear all plots
            for ax in self.axes.flat:
                if ax != self.axes[1, 2]:  # Don't clear stats panel
                    ax.clear()
            
            self.setup_plots()  # Reconfigure plots
            
            # Fix for Loss plot (axes[0, 0])
            has_loss_labels = False
            loss_series = self.ma_data['loss']
            if steps and loss_series:
                if self.phase_markers:
                    first_labeled = False
                    for i, (start_step, _) in enumerate(self.phase_markers):
                        end_step = self.phase_markers[i+1][0] - 1 if i + 1 < len(self.phase_markers) else steps[-1]
                        indices = [idx for idx, s in enumerate(steps) if s >= start_step and s <= end_step and not np.isnan(loss_series[idx])]
                        if indices:
                            x_seg = [steps[idx] for idx in indices]
                            y_seg = [loss_series[idx] for idx in indices]
                            self.axes[0, 0].plot(
                                x_seg,
                                y_seg,
                                'b-',
                                alpha=0.8,
                                linewidth=2,
                                label=(f'Loss (MA-{self.ma_window})' if not first_labeled else None)
                            )
                            first_labeled = True
                            has_loss_labels = True
                else:
                    valid_idx = [i for i, y in enumerate(loss_series) if not np.isnan(y)]
                    if valid_idx:
                        x_all = [steps[i] for i in valid_idx]
                        y_all = [loss_series[i] for i in valid_idx]
                        self.axes[0, 0].plot(x_all, y_all, 'b-', alpha=0.8, linewidth=2, label=f'Loss (MA-{self.ma_window})')
                        has_loss_labels = True
             
            # Also plot raw loss data as faint background
            valid_loss_raw = [(s, l) for s, l in zip(steps, self.data['loss']) if not np.isnan(l)]
            if valid_loss_raw:
                loss_steps_raw, loss_vals_raw = zip(*valid_loss_raw)
                self.axes[0, 0].plot(loss_steps_raw, loss_vals_raw, 'b-', alpha=0.2, linewidth=1, label='Raw Loss')
                has_loss_labels = True
 
            # Only add legend if there are labeled plots
            if has_loss_labels:
                self.axes[0, 0].legend()
 
            # Fix for Accuracy plot (axes[0, 1])
            has_acc_labels = False
            acc1_series = self.ma_data['acc1_selection']
            acc2_series = self.ma_data['acc2_selection']
            if steps and acc1_series:
                if self.phase_markers:
                    first_label_acc1 = False
                    for i, (start_step, _) in enumerate(self.phase_markers):
                        end_step = self.phase_markers[i+1][0] - 1 if i + 1 < len(self.phase_markers) else steps[-1]
                        indices = [idx for idx, s in enumerate(steps) if s >= start_step and s <= end_step and not np.isnan(acc1_series[idx])]
                        if indices:
                            x_seg = [steps[idx] for idx in indices]
                            y_seg = [acc1_series[idx] for idx in indices]
                            self.axes[0, 1].plot(
                                x_seg,
                                y_seg,
                                'g-',
                                label=(f'Agent1 (MA-{self.ma_window})' if not first_label_acc1 else None),
                                alpha=0.8,
                                linewidth=2
                            )
                            first_label_acc1 = True
                            has_acc_labels = True
                else:
                    valid_idx = [i for i, y in enumerate(acc1_series) if not np.isnan(y)]
                    if valid_idx:
                        x_all = [steps[i] for i in valid_idx]
                        y_all = [acc1_series[i] for i in valid_idx]
                        self.axes[0, 1].plot(x_all, y_all, 'g-', label=f'Agent1 (MA-{self.ma_window})', alpha=0.8, linewidth=2)
                        has_acc_labels = True

            if steps and acc2_series:
                if self.phase_markers:
                    first_label_acc2 = False
                    for i, (start_step, _) in enumerate(self.phase_markers):
                        end_step = self.phase_markers[i+1][0] - 1 if i + 1 < len(self.phase_markers) else steps[-1]
                        indices = [idx for idx, s in enumerate(steps) if s >= start_step and s <= end_step and not np.isnan(acc2_series[idx])]
                        if indices:
                            x_seg = [steps[idx] for idx in indices]
                            y_seg = [acc2_series[idx] for idx in indices]
                            self.axes[0, 1].plot(
                                x_seg,
                                y_seg,
                                'r-',
                                label=(f'Agent2 (MA-{self.ma_window})' if not first_label_acc2 else None),
                                alpha=0.8,
                                linewidth=2
                            )
                            first_label_acc2 = True
                            has_acc_labels = True
                else:
                    valid_idx = [i for i, y in enumerate(acc2_series) if not np.isnan(y)]
                    if valid_idx:
                        x_all = [steps[i] for i in valid_idx]
                        y_all = [acc2_series[i] for i in valid_idx]
                        self.axes[0, 1].plot(x_all, y_all, 'r-', label=f'Agent2 (MA-{self.ma_window})', alpha=0.8, linewidth=2)
                        has_acc_labels = True
 
            # Plot raw accuracy data as faint background
            valid_acc1_raw = [(s, a) for s, a in zip(steps, self.data['acc1_selection']) if not np.isnan(a)]
            valid_acc2_raw = [(s, a) for s, a in zip(steps, self.data['acc2_selection']) if not np.isnan(a)]
 
            if valid_acc1_raw:
                acc1_steps_raw, acc1_vals_raw = zip(*valid_acc1_raw)
                self.axes[0, 1].plot(acc1_steps_raw, acc1_vals_raw, 'g-', alpha=0.2, linewidth=1)
                 
            if valid_acc2_raw:
                acc2_steps_raw, acc2_vals_raw = zip(*valid_acc2_raw)
                self.axes[0, 1].plot(acc2_steps_raw, acc2_vals_raw, 'r-', alpha=0.2, linewidth=1)
 
            # Only add legend if there are labeled plots    
            if has_acc_labels:
                self.axes[0, 1].legend()
 
            # GES plot (axes[0, 2])
            has_ges_labels = False
            ges1_series = self.ma_data['ges1']
            ges2_series = self.ma_data['ges2']
            if steps and ges1_series:
                if self.phase_markers:
                    first_label_ges1 = False
                    for i, (start_step, _) in enumerate(self.phase_markers):
                        end_step = self.phase_markers[i+1][0] - 1 if i + 1 < len(self.phase_markers) else steps[-1]
                        indices = [idx for idx, s in enumerate(steps) if s >= start_step and s <= end_step and not np.isnan(ges1_series[idx])]
                        if indices:
                            x_seg = [steps[idx] for idx in indices]
                            y_seg = [ges1_series[idx] for idx in indices]
                            self.axes[0, 2].plot(
                                x_seg,
                                y_seg,
                                'g--',
                                label=(f'Agent1 GES (MA-{self.ma_window})' if not first_label_ges1 else None),
                                alpha=0.8,
                                linewidth=2
                            )
                            first_label_ges1 = True
                            has_ges_labels = True
                else:
                    valid_idx = [i for i, y in enumerate(ges1_series) if not np.isnan(y)]
                    if valid_idx:
                        x_all = [steps[i] for i in valid_idx]
                        y_all = [ges1_series[i] for i in valid_idx]
                        self.axes[0, 2].plot(x_all, y_all, 'g--', label=f'Agent1 GES (MA-{self.ma_window})', alpha=0.8, linewidth=2)
                        has_ges_labels = True

            if steps and ges2_series:
                if self.phase_markers:
                    first_label_ges2 = False
                    for i, (start_step, _) in enumerate(self.phase_markers):
                        end_step = self.phase_markers[i+1][0] - 1 if i + 1 < len(self.phase_markers) else steps[-1]
                        indices = [idx for idx, s in enumerate(steps) if s >= start_step and s <= end_step and not np.isnan(ges2_series[idx])]
                        if indices:
                            x_seg = [steps[idx] for idx in indices]
                            y_seg = [ges2_series[idx] for idx in indices]
                            self.axes[0, 2].plot(
                                x_seg,
                                y_seg,
                                'r--',
                                label=(f'Agent2 GES (MA-{self.ma_window})' if not first_label_ges2 else None),
                                alpha=0.8,
                                linewidth=2
                            )
                            first_label_ges2 = True
                            has_ges_labels = True
                else:
                    valid_idx = [i for i, y in enumerate(ges2_series) if not np.isnan(y)]
                    if valid_idx:
                        x_all = [steps[i] for i in valid_idx]
                        y_all = [ges2_series[i] for i in valid_idx]
                        self.axes[0, 2].plot(x_all, y_all, 'r--', label=f'Agent2 GES (MA-{self.ma_window})', alpha=0.8, linewidth=2)
                        has_ges_labels = True

            # Plot raw GES data as faint background
            valid_ges1_raw = [(s, c) for s, c in zip(steps, self.data['ges1']) if not np.isnan(c)]
            valid_ges2_raw = [(s, c) for s, c in zip(steps, self.data['ges2']) if not np.isnan(c)]

            if valid_ges1_raw:
                ges1_steps_raw, ges1_vals_raw = zip(*valid_ges1_raw)
                self.axes[0, 2].plot(ges1_steps_raw, ges1_vals_raw, 'g--', alpha=0.2, linewidth=1)
                 
            if valid_ges2_raw:
                ges2_steps_raw, ges2_vals_raw = zip(*valid_ges2_raw)
                self.axes[0, 2].plot(ges2_steps_raw, ges2_vals_raw, 'r--', alpha=0.2, linewidth=1)

            # Horizontal reference line at y=0
            self.axes[0, 2].axhline(y=0, color='gray', linestyle=':', linewidth=1)
 
            # Only add legend if there are labeled plots
            if has_ges_labels:
                self.axes[0, 2].legend()
            
            # Plot active symbols
            self.axes[1, 0].plot(steps, self.data['active_symbols'], 'purple', linewidth=2)
            
            # Set x-axis limits for all main plots to show all data (no sliding)
            if steps:
                x_min, x_max = steps[0], steps[-1]
                for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[0, 2], self.axes[1, 0]]:
                    ax.set_xlim(x_min, x_max)
            
            # Add phase markers to all relevant plots
            for step, phase in self.phase_markers:
                color = self.phase_colors.get(phase, 'gray')
                for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[0, 2], self.axes[1, 0]]:
                    ax.axvline(x=step, color=color, alpha=0.5, linestyle='--', linewidth=1)
            
            # Phase timeline
            if self.phase_markers:
                y_pos = 0
                for i, (step, phase) in enumerate(self.phase_markers):
                    color = self.phase_colors.get(phase, 'gray')
                    next_step = self.phase_markers[i+1][0] if i+1 < len(self.phase_markers) else steps[-1]
                    
                    rect = Rectangle((step, y_pos), next_step - step, 1, 
                                   facecolor=color, alpha=0.6, edgecolor='black')
                    self.axes[1, 1].add_patch(rect)
                    
                    # Add phase label
                    mid_step = (step + next_step) / 2
                    self.axes[1, 1].text(mid_step, y_pos + 0.5, phase.title(), 
                                       ha='center', va='center', fontsize=8, weight='bold')
                
                self.axes[1, 1].set_xlim(steps[0], steps[-1])
                self.axes[1, 1].set_ylim(-0.1, 1.1)
                self.axes[1, 1].set_yticks([0.5])
                self.axes[1, 1].set_yticklabels(['Phases'])
            
            # Update statistics panel
            self.axes[1, 2].clear()
            self.axes[1, 2].axis('off')
            
            if steps:
                recent_data = {k: [v for v in vals[-50:] if not np.isnan(v)] 
                             for k, vals in self.data.items() if k != 'steps'}
                
                stats_text = f"Current Step: {steps[-1]}\n\n"
                
                if self.ma_data['loss'] and not np.isnan(self.ma_data['loss'][-1]):
                    stats_text += f"Loss (MA): {self.ma_data['loss'][-1]:.4f}\n"
                if self.ma_data['acc1_selection'] and not np.isnan(self.ma_data['acc1_selection'][-1]):
                    stats_text += f"Agent1 Acc (MA): {self.ma_data['acc1_selection'][-1]:.3f}\n"
                if self.ma_data['acc2_selection'] and not np.isnan(self.ma_data['acc2_selection'][-1]):
                    stats_text += f"Agent2 Acc (MA): {self.ma_data['acc2_selection'][-1]:.3f}\n"
                if self.ma_data['ges1'] and not np.isnan(self.ma_data['ges1'][-1]):
                    stats_text += f"Agent1 GES (MA): {self.ma_data['ges1'][-1]:.2f}\n"
                if self.ma_data['ges2'] and not np.isnan(self.ma_data['ges2'][-1]):
                    stats_text += f"Agent2 GES (MA): {self.ma_data['ges2'][-1]:.2f}\n"
                if recent_data['active_symbols']:
                    stats_text += f"Active Symbols: {recent_data['active_symbols'][-1]}\n"
                if recent_data.get('total_puzzles'):
                    stats_text += f"Total Puzzles: {int(recent_data['total_puzzles'][-1])}\n"
                
                # Add phase info
                if self.phase_markers:
                    current_phase = self.phase_markers[-1][1]
                    stats_text += f"\nCurrent Phase: {current_phase.title()}\n"
                
                # Add raw recent values for comparison
                stats_text += f"\nRaw Recent Values:\n"
                if recent_data['loss']:
                    stats_text += f"Loss (raw): {recent_data['loss'][-1]:.4f}\n"
                if recent_data['acc1_selection']:
                    stats_text += f"Agent1 Acc (raw): {recent_data['acc1_selection'][-1]:.3f}\n"
                if recent_data['acc2_selection']:
                    stats_text += f"Agent2 Acc (raw): {recent_data['acc2_selection'][-1]:.3f}\n"
                if recent_data['ges1']:
                    stats_text += f"Agent1 GES (raw): {recent_data['ges1'][-1]:.2f}\n"
                if recent_data['ges2']:
                    stats_text += f"Agent2 GES (raw): {recent_data['ges2'][-1]:.2f}\n"
                
                self.axes[1, 2].text(0.05, 0.95, stats_text, fontsize=10, 
                                   verticalalignment='top', 
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor="lightblue", alpha=0.7))
            
            # Force update
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating live graph: {e}")
    
    def mark_phase_change(self, step, phase):
        """Mark a phase change in the graph"""
        with self.lock:
            self.phase_markers.append((step, phase))
            # Reset moving averages at the start of each new phase
            for ma in self.moving_averages.values():
                ma.values.clear()
            print(f"Phase change marked: Step {step} -> {phase} (moving averages reset)")
    
    def add_symbols(self, count):
        """Add symbols to the active count"""
        self.active_symbols_count += count
        print(f"Added {count} symbols, total active symbols: {self.active_symbols_count}")
    
    def remove_symbols(self, count):
        """Remove symbols from the active count"""
        self.active_symbols_count -= count
        print(f"Removed {count} symbols, total active symbols: {self.active_symbols_count}")
    
    def save_final_plot(self, filename="final_training_metrics.png"):
        """Save the final plot to file"""
        with self.lock:
            try:
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Final plot saved to {filename}")
            except Exception as e:
                print(f"Error saving final plot: {e}")
    
    def close(self):
        """Close the live grapher"""
        plt.ioff()
        plt.close(self.fig)

# Global live grapher instance
live_grapher = None
global_step_counter = 0

class MovingAverage:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def update(self, value):
        self.values.append(value)
        
    def get_average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

# Existing functions from train.py (unchanged)
def load_arc_puzzles(file_path):
    """Load all examples from ARC puzzles JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_examples = []
    for puzzle_id, puzzle_data in data.items():
        try:
            # Extract all training examples
            for train_example in puzzle_data['train']:
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(train_example['input']),
                        np.array(train_example['output'])
                    )
                )
                
                # Also use the output as an input example since we're testing communication
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(train_example['output']),
                        np.array(train_example['output'])
                    )
                )
            
            # Extract test examples
            for test_example in puzzle_data['test']:
                all_examples.append(
                    Puzzle.from_single_example(
                        np.array(test_example['input']),
                        np.array(test_example['input'])
                    )
                )
                
                if 'output' in test_example:
                    all_examples.append(
                        Puzzle.from_single_example(
                            np.array(test_example['output']),
                            np.array(test_example['output'])
                        )
                    )
                    
        except (ValueError, TypeError) as e:
            print(f"Skipping puzzle {puzzle_id} due to error: {e}")
            continue
            
    print(f"Extracted {len(all_examples)} total examples from {len(data)} puzzles")
    return all_examples

def run_pretraining_phase(trainer, target_puzzles=None, epochs=50):
    """
    Modified pretraining to only train on puzzles that have symbol mappings
    NOW USES GLOBAL INDICES CONSISTENTLY
    """
    global live_grapher
    
    print(f"\n{'='*60}")
    print(f"PRETRAINING PHASE - Encoder Training ({epochs} epochs)")
    print(f"{'='*60}")
    
    if target_puzzles is None:
        target_puzzles = trainer.active_puzzles
    
    agent1, agent2 = trainer.agent1, trainer.agent2
    device = trainer.device
    
    # Filter target puzzles to only include those with symbol mappings
    mapped_puzzles = []
    mapped_puzzle_indices = []
    
    for i, puzzle in enumerate(target_puzzles):
        # Find this puzzle's index in active_puzzles
        try:
            active_idx = trainer.active_puzzles.index(puzzle)
            if active_idx in trainer.puzzle_symbol_mapping:
                mapped_puzzles.append(puzzle)
                mapped_puzzle_indices.append(active_idx)
        except ValueError:
            # Puzzle not in active list - skip
            continue
    
    print(f"Training on {len(mapped_puzzles)} puzzles with symbol mappings")
    print(f"Skipping {len(target_puzzles) - len(mapped_puzzles)} puzzles without mappings")
    
    if len(mapped_puzzles) == 0:
        print("No puzzles with symbol mappings to train on!")
        return {'loss': [], 'accuracy': [], 'epochs': []}
    
    # Show current vocabulary state
    print(f"\nCurrent Vocabulary State:")
    print(f"  Agent1 communication symbols: {agent1.current_comm_symbols}")
    print(f"  Agent1 total symbols: {agent1.current_total_symbols}")
    print(f"  Puzzle symbols range: 0-{agent1.puzzle_symbols-1}")
    print(f"  Communication symbols range: {agent1.puzzle_symbols}-{agent1.current_total_symbols-1}")
    
    # Show existing symbol mappings
    print(f"\nCurrent Puzzle-Symbol Mappings:")
    for puzzle_idx, symbol_idx in trainer.puzzle_symbol_mapping.items():
        print(f"  Puzzle {puzzle_idx} → Symbol {symbol_idx}")
    
    # Convert puzzles to tensors and assign target symbols
    # NOW USING GLOBAL INDICES AS KEYS
    puzzle_tensors = {}  # global_idx -> tensor
    targets = {}         # global_idx -> target_symbol
    global_to_local = {} # global_idx -> position in processing order
    
    print(f"\nPuzzle-Symbol Assignments for Pretraining:")
    for local_pos, (puzzle, global_idx) in enumerate(zip(mapped_puzzles, mapped_puzzle_indices)):
        puzzle_tensor = torch.tensor(
            puzzle.test_input, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        # Store using global index
        puzzle_tensors[global_idx] = puzzle_tensor
        global_to_local[global_idx] = local_pos
        
        # Get the symbol mapping
        target_symbol = trainer.puzzle_symbol_mapping[global_idx]
        comm_symbol_idx = target_symbol - agent1.puzzle_symbols
        targets[global_idx] = comm_symbol_idx
        
        print(f"  Global Puzzle {global_idx}: Symbol {target_symbol} (Comm #{comm_symbol_idx})")
        print(f"    Grid shape: {puzzle_tensor.shape[1:]} - {puzzle_tensor[0].cpu().numpy()[:3, :3]}...")
    
    print(f"\nSymbol Assignment Summary:")
    print(f"  Puzzles with symbol mappings: {len(mapped_puzzles)}")
    print(f"  Global indices: {sorted(mapped_puzzle_indices)}")
    print(f"  Training mapping: {dict(sorted(targets.items()))}")
    
    # Show what we're training
    print(f"\nTraining Configuration:")
    print(f"  Encoder components: encoder, embedding_system, message_pooling")
    print(f"  Target symbols: {sorted(list(set(targets.values())))}")
    print(f"  Agent2 disabled during pretraining")
    
    # Training setup
    encoder_params = []
    component_counts = {'encoder': 0, 'embedding_system': 0, 'message_pooling': 0}
    
    for name, param in agent1.named_parameters():
        if any(component in name for component in ['encoder', 'embedding_system', 'message_pooling']):
            encoder_params.append(param)
            param.requires_grad = True
            
            # Count parameters by component
            for component in component_counts:
                if component in name:
                    component_counts[component] += param.numel()
        else:
            param.requires_grad = False
    
    print(f"\nTrainable Parameters:")
    for component, count in component_counts.items():
        print(f"  {component}: {count:,} parameters")
    print(f"  Total trainable: {sum(component_counts.values()):,} parameters")
    
    # Disable Agent2 during pretraining
    for param in agent2.parameters():
        param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=0.0001)
    
    history = {
        'loss': [],
        'accuracy': [],
        'epochs': []
    }
    
    visualization_frequency = max(1, epochs // 5)  # Show details 5 times during training
    
    # Mark phase change in live grapher
    global global_step_counter
    if live_grapher:
        live_grapher.mark_phase_change(global_step_counter, 'pretraining')
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        epoch_predictions = {}  # Track what each puzzle predicted
        
        # Create randomized order of global indices
        global_indices = list(mapped_puzzle_indices)
        random.shuffle(global_indices)
        
        show_details = (epoch % visualization_frequency == 0) or (epoch == epochs - 1)
        if show_details:
            print(f"\n--- Pretraining Epoch {epoch+1} Details ---")
        
        for global_idx in global_indices:
            puzzle_tensor = puzzle_tensors[global_idx]
            
            encoder_optimizer.zero_grad()
            
            # Forward pass through Agent1's encoder
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1, deterministic=True
            )
            
            pred_symbol = symbols[0, 0].argmax().item()
            target = torch.tensor([targets[global_idx]], device=device)
            
            # Store prediction for visualization
            epoch_predictions[global_idx] = pred_symbol
            
            # Symbol prediction loss
            symbol_loss = F.cross_entropy(symbol_logits[0, 0].unsqueeze(0), target)
            
            # Regularization
            reg_loss = 0.0
            for param in encoder_params:
                reg_loss += param.pow(2.0).sum()
            
            total_loss_item = symbol_loss + 0.001 * reg_loss
            
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            encoder_optimizer.step()
            
            # Track metrics
            if pred_symbol == targets[global_idx]:
                correct += 1
            total_loss += total_loss_item.item()
            
            # Show detailed learning for first few puzzles during visualization epochs
            if show_details and global_to_local[global_idx] < min(3, len(mapped_puzzles)):
                correct_symbol = "✓" if pred_symbol == targets[global_idx] else "✗"
                confidence = F.softmax(symbol_logits[0, 0], dim=0)[targets[global_idx]].item()
                print(f"  Global Puzzle {global_idx}: Target {targets[global_idx]} → Predicted {pred_symbol} {correct_symbol} (conf: {confidence:.3f})")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(mapped_puzzles)
        accuracy = correct / len(mapped_puzzles)
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['epochs'].append(epoch + 1)
        
        # Update live grapher
        if live_grapher:
            try:
                live_grapher.add_data_point(
                    step=global_step_counter,
                    loss=avg_loss,
                    acc1=accuracy,  # For pretraining, use accuracy as proxy
                    total_puzzles=len(trainer.active_puzzles)
                )
                global_step_counter += 1
            except Exception as e:
                print(f"Warning: Live grapher update failed: {e}")
        
        if show_details:
            print(f"  Epoch {epoch+1} Summary: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
            
            # Show symbol prediction distribution
            symbol_counts = {}
            for global_puzzle_idx, pred in epoch_predictions.items():
                if pred not in symbol_counts:
                    symbol_counts[pred] = []
                symbol_counts[pred].append(global_puzzle_idx)
            
            print(f"  Symbol predictions this epoch:")
            for symbol in sorted(symbol_counts.keys()):
                global_puzzles = symbol_counts[symbol]
                correct_count = sum(1 for p in global_puzzles if targets[p] == symbol)
                print(f"    Symbol {symbol}: {len(global_puzzles)} puzzles ({correct_count} correct) - global puzzles {sorted(global_puzzles)}")
        
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Pretraining Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
    
    # Final symbol assignment verification
    print(f"\n{'='*40}")
    print(f"PRETRAINING VERIFICATION")
    print(f"{'='*40}")
    
    agent1.eval()
    with torch.no_grad():
        print(f"Final symbol assignments (verification):")
        for global_idx in sorted(mapped_puzzle_indices):
            puzzle_tensor = puzzle_tensors[global_idx]
            symbols, symbol_logits, _ = agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1, deterministic=True
            )
            pred_symbol = symbols[0, 0].argmax().item()
            target_symbol = targets[global_idx]
            confidence = F.softmax(symbol_logits[0, 0], dim=0)[target_symbol].item()
            
            status = "✓" if pred_symbol == target_symbol else "✗"
            print(f"  Global Puzzle {global_idx}: Target {target_symbol} → Final {pred_symbol} {status} (conf: {confidence:.3f})")
    
    agent1.train()
    
    # Copy Agent1's weights to Agent2
    print(f"\n{'='*40}")
    print(f"COPYING WEIGHTS Agent1 → Agent2")
    print(f"{'='*40}")
    
    with torch.no_grad():
        total_copied = 0
        
        # Copy encoder
        encoder_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.encoder.named_parameters(), agent2.encoder.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                encoder_copied += param1.numel()
        print(f"  ✓ Encoder: {encoder_copied:,} parameters")
        total_copied += encoder_copied
        
        # Copy embedding system
        embedding_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.embedding_system.named_parameters(), agent2.embedding_system.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                embedding_copied += param1.numel()
        print(f"  ✓ Embedding system: {embedding_copied:,} parameters")
        total_copied += embedding_copied
        
        # Copy message pooling
        pooling_copied = 0
        for (name1, param1), (name2, param2) in zip(agent1.message_pooling.named_parameters(), agent2.message_pooling.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                pooling_copied += param1.numel()
        print(f"  ✓ Message pooling: {pooling_copied:,} parameters")
        total_copied += pooling_copied
        
        # Copy communication embeddings
        start_idx = agent1.puzzle_symbols
        end_idx = start_idx + agent1.current_comm_symbols
        agent2.communication_embedding.weight[start_idx:end_idx].copy_(
            agent1.communication_embedding.weight[start_idx:end_idx]
        )
        comm_copied = agent1.current_comm_symbols * agent1.communication_embedding.embedding_dim
        print(f"  ✓ Communication embeddings: {comm_copied:,} parameters (symbols {start_idx}-{end_idx-1})")
        total_copied += comm_copied
        
        print(f"Total parameters copied: {total_copied:,}")
    
    # Re-enable all parameters
    for param in agent1.parameters():
        param.requires_grad = True
    for param in agent2.parameters():
        param.requires_grad = True
    
    print(f"\n{'='*40}")
    print(f"PRETRAINING COMPLETE")
    print(f"{'='*40}")
    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Trained on {len(mapped_puzzles)} puzzles with symbol mappings")
    print(f"Global puzzle indices: {sorted(mapped_puzzle_indices)}")
    print(f"Both agents now have synchronized encoders")
    
    return history

def run_training_phase(trainer, cycles=200):
    """Run the main training phase - MODIFIED to repeat each puzzle multiple times"""
    global live_grapher
    
    print(f"\n{'='*60}")
    print(f"TRAINING PHASE - Joint Training ({cycles} cycles)")
    print(f"Repetitions per puzzle: {trainer.repetitions_per_puzzle}")
    print(f"{'='*60}")
    
    trainer.set_training_mode("joint")
    
    # Mark phase change in live grapher
    global global_step_counter
    if live_grapher:
        live_grapher.mark_phase_change(global_step_counter, 'training')
    
    # Initialize tracking
    metrics_history = []
    acc1_selection_history = []
    acc2_selection_history = []
    ges1_history = []
    ges2_history = []
    
    # Moving averages
    ma_window = 50
    acc1_selection_ma = MovingAverage(ma_window)
    acc2_selection_ma = MovingAverage(ma_window)
    ges1_ma = MovingAverage(ma_window)
    ges2_ma = MovingAverage(ma_window)
    
    for cycle in range(cycles):
        print(f"\nTraining Cycle {cycle + 1}/{cycles}")
        
        # MODIFIED: Train on each active puzzle multiple times before moving to next
        cycle_metrics = []
        for puzzle_idx, puzzle in enumerate(trainer.active_puzzles):
            puzzle_tensor = torch.tensor(
                puzzle.test_input, 
                dtype=torch.long, 
                device=trainer.device
            ).unsqueeze(0)
            
            print(f"  Training on puzzle {puzzle_idx} for {trainer.repetitions_per_puzzle} repetitions...")
            
            # NEW: Repeat each puzzle multiple times
            for repetition in range(trainer.repetitions_per_puzzle):
                step_metrics = trainer.train_bidirectional_step(
                    puzzle_tensor, 
                    puzzle_idx,
                    num_exchanges=1,
                    temperature=1.0,
                    initial_phase=False
                )
                
                cycle_metrics.extend(step_metrics)
                
                # Update metrics for each step within the repetition
                for metrics in step_metrics:
                    # Update accuracy moving averages
                    acc1_selection_ma.update(metrics['agent1_selection_accuracy'])
                    acc2_selection_ma.update(metrics['agent2_selection_accuracy'])

                    # Compute GES for both agents
                    try:
                        chance = 1.0 / max(1, metrics.get('num_candidates', trainer.num_distractors + 1))
                        active_puzzles = max(1, metrics.get('active_puzzles', len(trainer.active_puzzles)))
                        symbols = metrics.get('mapped_puzzles', len(trainer.puzzle_symbol_mapping))
                        ratio = (active_puzzles / symbols) if symbols and symbols > 0 else np.nan
                        ges1_val = ((metrics['agent1_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else np.nan
                        ges2_val = ((metrics['agent2_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else np.nan
                    except Exception:
                        ges1_val, ges2_val = np.nan, np.nan
                    
                    # Update GES moving averages
                    if not np.isnan(ges1_val):
                        ges1_ma.update(ges1_val)
                    if not np.isnan(ges2_val):
                        ges2_ma.update(ges2_val)
                    
                    acc1_selection_history.append(acc1_selection_ma.get_average())
                    acc2_selection_history.append(acc2_selection_ma.get_average())
                    ges1_history.append(ges1_ma.get_average())
                    ges2_history.append(ges2_ma.get_average())
                    
                    # Update live grapher for each step
                    if live_grapher:
                        try:
                            live_grapher.add_data_point(
                                step=global_step_counter,
                                loss=metrics['total_loss'],
                                acc1=metrics['agent1_selection_accuracy'],
                                acc2=metrics['agent2_selection_accuracy'],
                                ges1=ges1_val,
                                ges2=ges2_val,
                                total_puzzles=metrics.get('active_puzzles', len(trainer.active_puzzles))
                            )
                            global_step_counter += 1
                        except Exception as e:
                            print(f"Warning: Live grapher update failed: {e}")
                
                # Show progress for this repetition
                if step_metrics:
                    avg_loss = step_metrics[0]['total_loss']
                    acc1 = step_metrics[0]['agent1_selection_accuracy']
                    acc2 = step_metrics[0]['agent2_selection_accuracy']
                    print(f"    Rep {repetition+1}/{trainer.repetitions_per_puzzle}: Loss={avg_loss:.4f}, Acc1={acc1:.3f}, Acc2={acc2:.3f}")
        
        # Update comprehensive metrics history
        metrics_history.extend(cycle_metrics)
        
        # Show cycle summary
        if cycle_metrics:
            avg_acc1 = acc1_selection_ma.get_average()
            avg_acc2 = acc2_selection_ma.get_average()
            avg_loss = np.mean([m['total_loss'] for m in cycle_metrics if not np.isnan(m['total_loss'])])
            print(f"  Cycle {cycle + 1} Summary: Avg Loss: {avg_loss:.4f}, Acc1: {avg_acc1:.3f}, Acc2: {avg_acc2:.3f}")
            print(f"  Total training steps this cycle: {len(cycle_metrics)}")
    
    accuracies_history = {
        'acc1_selection': acc1_selection_history,
        'acc2_selection': acc2_selection_history,
        'ges1': ges1_history,
        'ges2': ges2_history
    }
    
    print(f"\nTraining Phase Complete:")
    print(f"  Total training steps: {len(metrics_history)}")
    print(f"  Steps per cycle: {len(trainer.active_puzzles) * trainer.repetitions_per_puzzle}")
    
    return metrics_history, accuracies_history

def run_consolidation_phase(trainer):
    """Run consolidation phase to remove recessive symbols"""
    global live_grapher
    
    print(f"\n{'='*60}")
    print(f"CONSOLIDATION PHASE")
    print(f"{'='*60}")
    
    # Mark phase change in live grapher
    global global_step_counter
    if live_grapher:
        live_grapher.mark_phase_change(global_step_counter, 'consolidation')
    
    # Run consolidation tests
    confusion_data = trainer.run_consolidation_test()
    
    # Identify recessive symbols
    recessive_symbols = trainer.identify_recessive_symbols(confusion_data)
    
    # Remove recessive symbols
    trainer.remove_recessive_symbols(recessive_symbols)
    
    return confusion_data, recessive_symbols

# --- NEW: Remedial training phase ---
def run_remedial_phase(trainer, accuracy_threshold=0.7, tests=10, train_cycles_per_round=20, max_rounds=10):
    """Train only weak puzzles (<70% accuracy over 10 tests) until each reaches threshold.
    Alternates between short training rounds and re-testing. Returns metrics and accuracies history.
    """
    global live_grapher
    global global_step_counter

    print(f"\n{'='*60}")
    print(f"REMEDIAL PHASE")
    print(f"{'='*60}")

    # Mark phase change in live grapher
    if live_grapher:
        live_grapher.mark_phase_change(global_step_counter, 'remedial')

    # Identify initial weak puzzles
    threshold_correct = int(np.ceil(accuracy_threshold * tests))
    weak_indices = trainer.get_weak_puzzles(accuracy_threshold=accuracy_threshold, tests=tests)
    if not weak_indices:
        print("No remedial training needed (all puzzles >= threshold)")
        return [], {'acc1_selection': [], 'acc2_selection': [], 'ges1': [], 'ges2': []}

    print(f"Starting remedial training for puzzles: {weak_indices} (target >= {threshold_correct}/{tests})")

    metrics_history = []
    acc1_history, acc2_history = [], []
    ges1_history, ges2_history = [], []

    # Moving averages for live view
    ma_window = 50
    acc1_ma = MovingAverage(ma_window)
    acc2_ma = MovingAverage(ma_window)
    ges1_ma = MovingAverage(ma_window)
    ges2_ma = MovingAverage(ma_window)

    round_idx = 0
    while weak_indices and round_idx < max_rounds:
        round_idx += 1
        print(f"\nRemedial Round {round_idx}: Training {len(weak_indices)} puzzles for {train_cycles_per_round} cycles")
        for cycle in range(train_cycles_per_round):
            # Train each weak puzzle once per cycle
            for puzzle_idx in list(weak_indices):
                puzzle = trainer.active_puzzles[puzzle_idx]
                puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=trainer.device).unsqueeze(0)
                step_metrics = trainer.train_bidirectional_step(
                    puzzle_tensor,
                    puzzle_idx,
                    num_exchanges=1,
                    temperature=1.0,
                    initial_phase=False
                )
                metrics_history.extend(step_metrics)

                # Update live grapher for each step
                for metrics in step_metrics:
                    # Update accuracy MA
                    acc1_ma.update(metrics['agent1_selection_accuracy'])
                    acc2_ma.update(metrics['agent2_selection_accuracy'])

                    # Compute GES
                    try:
                        chance = 1.0 / max(1, metrics.get('num_candidates', trainer.num_distractors + 1))
                        active_puzzles = max(1, metrics.get('active_puzzles', len(trainer.active_puzzles)))
                        symbols = metrics.get('mapped_puzzles', len(trainer.puzzle_symbol_mapping))
                        ratio = (active_puzzles / symbols) if symbols and symbols > 0 else np.nan
                        ges1_val = ((metrics['agent1_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else np.nan
                        ges2_val = ((metrics['agent2_selection_accuracy'] - chance) * ratio * 100.0) if not np.isnan(ratio) else np.nan
                    except Exception:
                        ges1_val, ges2_val = np.nan, np.nan

                    if not np.isnan(ges1_val):
                        ges1_ma.update(ges1_val)
                    if not np.isnan(ges2_val):
                        ges2_ma.update(ges2_val)

                    acc1_history.append(acc1_ma.get_average())
                    acc2_history.append(acc2_ma.get_average())
                    ges1_history.append(ges1_ma.get_average())
                    ges2_history.append(ges2_ma.get_average())

                    if live_grapher:
                        try:
                            live_grapher.add_data_point(
                                step=global_step_counter,
                                loss=metrics['total_loss'],
                                acc1=metrics['agent1_selection_accuracy'],
                                acc2=metrics['agent2_selection_accuracy'],
                                ges1=ges1_val,
                                ges2=ges2_val,
                                total_puzzles=metrics.get('active_puzzles', len(trainer.active_puzzles))
                            )
                            global_step_counter += 1
                        except Exception as e:
                            print(f"Warning: Live grapher update failed: {e}")

        # Re-test only current weak puzzles
        print("\nRe-testing weak puzzles...")
        retest_results = trainer.evaluate_selection_accuracy(puzzle_indices=weak_indices, tests=tests)
        passed = [idx for idx, correct in retest_results.items() if correct >= threshold_correct]
        if passed:
            print(f"Puzzles reaching threshold this round: {passed}")
            # Remove passed puzzles from weak set
            weak_indices = [idx for idx in weak_indices if idx not in passed]
        else:
            print("No puzzles reached threshold this round.")

        print(f"Remaining weak puzzles: {weak_indices}")

    if weak_indices:
        print(f"Remedial phase ended with {len(weak_indices)} puzzles still below threshold after {round_idx} rounds.")
    else:
        print("All remedial puzzles reached the target accuracy.")

    accuracies_history = {
        'acc1_selection': acc1_history,
        'acc2_selection': acc2_history,
        'ges1': ges1_history,
        'ges2': ges2_history
    }

    return metrics_history, accuracies_history

def run_addition_phase(trainer):
    """Run addition phase to add new puzzles"""
    global live_grapher
    
    print(f"\n{'='*60}")
    print(f"ADDITION PHASE")
    print(f"{'='*60}")
    
    # Mark phase change in live grapher
    global global_step_counter
    if live_grapher:
        live_grapher.mark_phase_change(global_step_counter, 'addition')
    
    new_puzzles = trainer.add_new_puzzles()
    return new_puzzles

# Updated plotting function for phase-based training
def plot_phase_training_metrics(metrics_history, accuracies_history, phase_info, title="Phase-Based Training Metrics"):
    """Plot training metrics for phase-based training"""
    plt.figure(figsize=(20, 12))
    
    # Separate metrics by phase
    phases = ['pretraining', 'training', 'consolidation', 'addition', 'remedial']
    phase_colors = {'pretraining': 'blue', 'training': 'green', 'consolidation': 'orange', 'addition': 'red', 'remedial': 'magenta'}
    
    # Plot loss
    plt.subplot(3, 2, 1)
    losses = [m['total_loss'] for m in metrics_history if not np.isnan(m['total_loss'])]
    phase_markers = [m.get('phase', 'training') for m in metrics_history if not np.isnan(m['total_loss'])]
    
    current_phase = None
    start_idx = 0
    
    for i, phase in enumerate(phase_markers + ['END']):
        if phase != current_phase or i == len(phase_markers):
            if current_phase is not None and start_idx < i:
                plt.plot(range(start_idx, i), losses[start_idx:i], 
                        label=f'{current_phase.title()} Loss', 
                        color=phase_colors.get(current_phase, 'gray'),
                        alpha=0.7)
            current_phase = phase
            start_idx = i
    
    plt.title(f'{title} - Loss by Phase')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot selection accuracies
    plt.subplot(3, 2, 2)
    plt.plot(accuracies_history['acc1_selection'], label='Agent1 Selection Acc', alpha=0.8, linewidth=2)
    plt.plot(accuracies_history['acc2_selection'], label='Agent2 Selection Acc', alpha=0.8, linewidth=2)
    plt.title(f'{title} - Selection Accuracies')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot GES (replaces confidence)
    plt.subplot(3, 2, 3)
    # Moving averages over time captured in accuracies_history
    plt.plot(accuracies_history.get('ges1', []), label='Agent1 GES (MA)', alpha=0.7, linestyle='--', color='g')
    plt.plot(accuracies_history.get('ges2', []), label='Agent2 GES (MA)', alpha=0.7, linestyle='--', color='r')
    
    # Also plot raw GES computed from metrics_history as faint background
    raw_ges1, raw_ges2 = [], []
    for m in metrics_history:
        try:
            num_candidates = m.get('num_candidates', None)
            if num_candidates is None or num_candidates <= 0:
                continue
            chance = 1.0 / num_candidates
            puzzles = m.get('active_puzzles', 0)
            symbols = m.get('mapped_puzzles', 0)
            if symbols and symbols > 0:
                ratio = puzzles / symbols
                raw_ges1.append(((m['agent1_selection_accuracy'] - chance) * ratio * 100.0))
                raw_ges2.append(((m['agent2_selection_accuracy'] - chance) * ratio * 100.0))
            else:
                raw_ges1.append(np.nan)
                raw_ges2.append(np.nan)
        except Exception:
            raw_ges1.append(np.nan)
            raw_ges2.append(np.nan)
    if raw_ges1:
        plt.plot(raw_ges1, label='Agent1 GES (raw)', alpha=0.2, linewidth=1, color='g')
    if raw_ges2:
        plt.plot(raw_ges2, label='Agent2 GES (raw)', alpha=0.2, linewidth=1, color='r')
    
    plt.title(f'{title} - Generalization Efficiency Score (GES)')
    plt.xlabel('Step')
    plt.ylabel('GES')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    plt.grid(True)
    plt.legend()
    
    # Active symbols over time
    plt.subplot(3, 2, 4)
    active_symbols = [m.get('active_symbols', 0) for m in metrics_history]
    global_phases = [m.get('global_phase_count', 0) for m in metrics_history]
    
    plt.plot(active_symbols, label='Active Symbols', linewidth=2, color='purple')
    plt.title('Active Symbols Evolution')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    
    # Phase information summary
    plt.subplot(3, 2, 5)
    phase_text = f"Current Phase: {phase_info.get('current_phase', 'unknown').title()}\n"
    phase_text += f"Phase Cycle: {phase_info.get('phase_cycle', 0)}\n"
    phase_text += f"Global Phase Count: {phase_info.get('global_phase_count', 0)}\n"
    phase_text += f"Active Puzzles: {phase_info.get('active_puzzles', 0)}\n"
    phase_text += f"Removed Symbols: {phase_info.get('removed_symbols', 0)}\n\n"
    
    phase_text += "Phase Cycle:\n"
    phase_text += "1. Pretraining (new puzzles)\n"
    phase_text += "2. Training (all puzzles)\n"
    phase_text += "3. Consolidation (remove recessive)\n"
    phase_text += "4. Remedial (train weak puzzles)\n"
    phase_text += "5. Addition (add new puzzles)\n"
    
    plt.text(0.1, 0.9, phase_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Phase Information')
    
    # Training configuration
    plt.subplot(3, 2, 6)
    config = phase_info.get('selection_config', {})
    config_text = f"Training Configuration:\n\n"
    config_text += f"Num Distractors: {config.get('num_distractors', 'N/A')}\n"
    config_text += f"Distractor Strategy: {config.get('distractor_strategy', 'N/A')}\n"
    config_text += f"Training Cycles: {config.get('training_cycles', 'N/A')}\n"
    config_text += f"Consolidation Tests: {config.get('consolidation_tests', 'N/A')}\n"
    config_text += f"Puzzles per Addition: {config.get('puzzles_per_addition', 'N/A')}\n"
    config_text += f"Repetitions per Puzzle: {config.get('repetitions_per_puzzle', 'N/A')}\n"
    
    plt.text(0.1, 0.9, config_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Configuration')
    
    plt.tight_layout()
    plt.savefig('phase_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_selection_debug(puzzle_tensor, sender, receiver, trainer):
    """Debug function for selection task - updated for phase training"""
    print("\n  Phase-Based Selection Debug:")
    print_grid(puzzle_tensor[0], "Target Puzzle")
    
    # Show phase and vocabulary status
    phase_info = trainer.get_phase_status()
    print(f"\nCurrent Phase: {phase_info['current_phase']}")
    print(f"Phase Cycle: {phase_info['phase_cycle']}")
    print(f"Active Puzzles: {phase_info['active_puzzles']}")
    print(f"Removed Symbols: {phase_info['removed_symbols']}")
    print(f"Repetitions per Puzzle: {phase_info['selection_config']['repetitions_per_puzzle']}")
    
    # Show current puzzle-symbol mapping
    print(f"\nPuzzle-Symbol Mapping:")
    for puzzle_idx, symbol_idx in trainer.puzzle_symbol_mapping.items():
        print(f"  Puzzle {puzzle_idx} → Symbol {symbol_idx}")
    
    # Sender encodes message
    print(f"\nSender → Receiver Selection:")
    symbols, symbol_logits, stats = sender.encode_puzzle_to_message(puzzle_tensor, temperature=0.1)
    print_message_details(symbols, "Sender")
    
    # Create selection candidates from active puzzles
    candidates = []
    for puzzle in trainer.active_puzzles[:min(4, len(trainer.active_puzzles))]:  # Limit for display
        candidate_tensor = torch.tensor(
            puzzle.test_input, 
            dtype=torch.long, 
            device=puzzle_tensor.device
        ).unsqueeze(0)
        candidates.append(candidate_tensor)
    
    # Receiver selects
    selection_probs, selection_logits, debug_info = receiver.select_from_candidates(
        symbols, candidates, temperature=0.1
    )
    
    # Show results
    predicted_idx = selection_logits.argmax(dim=-1).item()
    target_confidence = selection_probs[0, 0].item()
    
    print(f"\nReceiver Selection Results:")
    print(f"  Number of candidates: {len(candidates)}")
    print(f"  Predicted choice: {predicted_idx} (target is 0)")
    print(f"  Selection correct: {'✓' if predicted_idx == 0 else '✗'}")
    print(f"  Confidence in target: {target_confidence:.4f}")
    
    print(f"\nAll selection probabilities:")
    for i, prob in enumerate(selection_probs[0]):
        marker = " ← target" if i == 0 else f" ← candidate {i}"
        symbol = "✓" if i == predicted_idx else " "
        print(f"    {symbol} Candidate {i}: {prob.item():.4f}{marker}")

def print_grid(grid: torch.Tensor, title: str = "Grid"):
    """Print a grid in a readable format"""
    print(f"\n{title}:")
    for row in grid.cpu().numpy():
        print("  " + " ".join(f"{x:2d}" for x in row))

def print_message_details(symbols: torch.Tensor, agent_name: str):
    """Print detailed information about the message being sent."""
    if len(symbols.shape) == 3:  # [batch, seq, num_symbols]
        message_indices = torch.argmax(symbols, dim=-1)[0]
    else:
        message_indices = symbols[0]
        
    nonzero_indices = message_indices[message_indices != 0]
    
    print(f"\n{agent_name} Message:")
    print(f"  Full sequence: {message_indices.tolist()}")
    print(f"  Non-zero symbols: {nonzero_indices.tolist()}")
    print(f"  Length: {len(message_indices)}")
    print(f"  Active symbols: {len(nonzero_indices)}")

def main():
    global live_grapher
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize live grapher with initial symbol count (will be set properly after trainer creation)
    print("Initializing live grapher...")
    live_grapher = LiveGrapher(initial_symbols=3)  # Temporary, will be updated after trainer setup
    print("Live grapher ready! Training metrics will update in real-time.")
    
    # Create selection agents
    sender = Agent(
        agent_id="sender",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=100,     
        puzzle_symbols=10,
        max_seq_length=2,    
        sender_scale=1.0,
        similarity_metric='cosine'
    ).to(device)
    
    receiver = Agent(
        agent_id="receiver",
        embedding_dim=512,
        hidden_dim=1024,
        num_symbols=100,     
        puzzle_symbols=10,
        max_seq_length=2,    
        sender_scale=1.0,
        similarity_metric='cosine'
    ).to(device)
    
    # Create phase-based trainer with all configurable parameters
    trainer = CommunicationTrainer(
        agent1=sender,
        agent2=receiver,
        learning_rate=7e-7,
        device=device,
        sync_frequency=1000000000,
        num_distractors=2,
        distractor_strategy='random',
        first_training_cycles=50,        # Shorter first training phase
        training_cycles=25,              # Subsequent training phases
        consolidation_tests=5,           # Test rounds in consolidation
        puzzles_per_addition=2,          # Puzzles to add each cycle
        repetitions_per_puzzle=1,        # Repeat each puzzle X times
        initial_puzzle_count=3,          # NEW: Initial number of puzzles
        initial_comm_symbols=3           # NEW: Initial communication symbols (optional, defaults to initial_puzzle_count)
    )
    max_global_phases = 20  # Run 3 complete cycles for demonstration

    first_pretrain_epochs = 100
    pretrain_epochs = 100
    
    # Load ARC puzzles
    arc_file_path = 'arc-agi_test_challenges.json'
    all_puzzles = load_arc_puzzles(arc_file_path)
    print(f"\nLoaded {len(all_puzzles)} total examples from ARC dataset")
    
    # Set puzzle dataset and initialize first puzzles
    trainer.set_puzzle_dataset(all_puzzles)
    trainer.initialize_first_puzzles()  # NEW: No parameter needed, uses trainer's configuration
    
    # Update live grapher with correct initial symbol count
    if live_grapher:
        initial_symbols = trainer.initial_comm_symbols
        live_grapher.active_symbols_count = initial_symbols
        print(f"Live grapher updated with initial symbols: {initial_symbols}")
    
    # Show initial state
    print("\n" + "="*60)
    print("INITIAL STATE")
    print("="*60)
    phase_info = trainer.get_phase_status()
    print(f"Phase: {phase_info['current_phase']}")
    print(f"Active puzzles: {phase_info['active_puzzles']}")
    print(f"Initial communication symbols: {phase_info['selection_config']['initial_comm_symbols']}")
    print(f"Repetitions per puzzle: {phase_info['selection_config']['repetitions_per_puzzle']}")
    print(f"Puzzle-symbol mapping: {phase_info['puzzle_symbol_mapping']}")
    sender.print_position_symbol_mapping()
    print("="*60)
    
    # Enhanced logging
    with open('phase_training_log.txt', 'w') as log_file:
        log_file.write("Phase-Based Training Log with Puzzle Repetitions\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Phase cycle: pretraining → training → consolidation → remedial → addition\n")
        log_file.write(f"Training cycles per phase: {trainer.training_cycles}\n")
        log_file.write(f"Repetitions per puzzle: {trainer.repetitions_per_puzzle}\n")
        log_file.write(f"Consolidation tests: {trainer.consolidation_tests}\n")
        log_file.write(f"Puzzles per addition: {trainer.puzzles_per_addition}\n")
        log_file.write("="*50 + "\n\n")
        
        # Initialize comprehensive tracking
        all_metrics_history = []
        all_accuracies_history = {
            'acc1_selection': [],
            'acc2_selection': [],
            'ges1': [],
            'ges2': []
        }
        
        
        # Note: For the final global phase, we skip consolidation and addition phases
        # to get accurate final counts after the last training phase
        while trainer.global_phase_count < max_global_phases:
            phase_info = trainer.get_phase_status()
            current_phase = phase_info['current_phase']
            
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"GLOBAL PHASE {trainer.global_phase_count + 1}/{max_global_phases} - {current_phase.upper()}\n")
            if trainer.global_phase_count == max_global_phases - 1:
                log_file.write(f"*** FINAL GLOBAL PHASE - Will skip consolidation/addition after training ***\n")
            log_file.write(f"{'='*60}\n")
            log_file.flush()
            
            if current_phase == "pretraining":
                # Pretraining phase - train encoder on newly added puzzles
                if trainer.global_phase_count == 0:
                    # First pretraining - use all initial puzzles
                    pretraining_history = run_pretraining_phase(trainer, epochs=first_pretrain_epochs)
                else:
                    # Subsequent pretraining - use only newly added puzzles
                    # Get last 5 puzzles (newly added)
                    new_puzzles = trainer.active_puzzles[-trainer.puzzles_per_addition:]
                    pretraining_history = run_pretraining_phase(trainer, target_puzzles=new_puzzles, epochs=pretrain_epochs)
                
                trainer.advance_phase()
                
            elif current_phase == "training":
                # Training phase - use different cycle counts for first vs subsequent phases
                cycles_for_this_phase = trainer.get_training_cycles_for_current_phase()
                
                # Log which type of training phase this is
                if trainer.global_phase_count == 0:
                    log_file.write(f"FIRST training phase - using {cycles_for_this_phase} cycles\n")
                    log_file.write(f"Each puzzle repeated {trainer.repetitions_per_puzzle} times per cycle\n")
                    print(f"FIRST training phase - using {cycles_for_this_phase} cycles")
                    print(f"Each puzzle repeated {trainer.repetitions_per_puzzle} times per cycle")
                else:
                    log_file.write(f"Subsequent training phase #{trainer.global_phase_count} - using {cycles_for_this_phase} cycles\n")
                    log_file.write(f"Each puzzle repeated {trainer.repetitions_per_puzzle} times per cycle\n")
                    print(f"Subsequent training phase #{trainer.global_phase_count} - using {cycles_for_this_phase} cycles")
                    print(f"Each puzzle repeated {trainer.repetitions_per_puzzle} times per cycle")
                
                training_metrics, training_accuracies = run_training_phase(trainer, cycles=cycles_for_this_phase)
                
                # Add to comprehensive tracking
                all_metrics_history.extend(training_metrics)
                for key in all_accuracies_history:
                    all_accuracies_history[key].extend(training_accuracies[key])
                
                # Log training summary
                if training_metrics:
                    final_acc1 = training_accuracies['acc1_selection'][-1] if training_accuracies['acc1_selection'] else 0
                    final_acc2 = training_accuracies['acc2_selection'][-1] if training_accuracies['acc2_selection'] else 0
                    avg_loss = np.mean([m['total_loss'] for m in training_metrics[-50:] if not np.isnan(m['total_loss'])])
                    
                    log_file.write(f"Training completed ({cycles_for_this_phase} cycles, {trainer.repetitions_per_puzzle} reps/puzzle):\n")
                    log_file.write(f"  Final Agent1 accuracy: {final_acc1:.3f}\n")
                    log_file.write(f"  Final Agent2 accuracy: {final_acc2:.3f}\n")
                    log_file.write(f"  Average loss (last 50): {avg_loss:.4f}\n")
                    log_file.write(f"  Total training steps: {len(training_metrics)}\n")
                    log_file.flush()
                
                # Check if this is the final global phase
                if trainer.global_phase_count == max_global_phases - 1:
                    log_file.write(f"Final training phase completed - skipping consolidation and addition phases\n")
                    log_file.flush()
                    
                    # Plot final training metrics before exiting
                    if all_metrics_history:
                        plot_phase_training_metrics(
                            all_metrics_history, 
                            all_accuracies_history,
                            trainer.get_phase_status(),
                            title=f"Phase-Based Training (Final - Global Phase {trainer.global_phase_count + 1})"
                        )
                        print("Final training metrics plotted to phase_training_metrics.png")
                    
                    break  # Exit the phase loop to prevent further phases
                
                trainer.advance_phase()
                
            elif current_phase == "consolidation":
                # Skip consolidation if this would be the final global phase
                if trainer.global_phase_count >= max_global_phases - 1:
                    log_file.write(f"Skipping consolidation phase for final global phase\n")
                    log_file.flush()
                    break
                
                # Consolidation phase - test and remove recessive symbols
                confusion_data, removed_symbols = run_consolidation_phase(trainer)
                
                # Update live grapher symbol count
                if live_grapher and removed_symbols:
                    live_grapher.remove_symbols(len(removed_symbols))
                
                log_file.write(f"Consolidation completed:\n")
                log_file.write(f"  Tested symbols: {len(confusion_data)}\n")
                log_file.write(f"  Removed symbols: {len(removed_symbols)}\n")
                log_file.write(f"  Remaining puzzles: {len(trainer.active_puzzles)}\n")
                if removed_symbols:
                    log_file.write(f"  Removed: {removed_symbols}\n")
                log_file.flush()
                
                trainer.advance_phase()
                
            elif current_phase == "remedial":
                # Remedial training phase - train only weak puzzles until >= 7/10
                log_file.write(f"Remedial phase starting: identifying weak puzzles (<7/10)\n")
                log_file.flush()

                remedial_metrics, remedial_accuracies = run_remedial_phase(
                    trainer,
                    accuracy_threshold=0.7,
                    tests=10,
                    train_cycles_per_round=20,
                    max_rounds=10
                )

                # Track
                all_metrics_history.extend(remedial_metrics)
                for key in all_accuracies_history:
                    all_accuracies_history[key].extend(remedial_accuracies[key])

                log_file.write(f"Remedial phase completed: steps={len(remedial_metrics)}\n")
                log_file.flush()

                trainer.advance_phase()
                
            elif current_phase == "addition":
                # Skip addition if this would be the final global phase
                if trainer.global_phase_count >= max_global_phases - 1:
                    log_file.write(f"Skipping addition phase for final global phase\n")
                    log_file.flush()
                    break
                
                # Addition phase - add new puzzles
                new_puzzles = run_addition_phase(trainer)
                
                # Update live grapher symbol count
                if live_grapher and new_puzzles:
                    live_grapher.add_symbols(len(new_puzzles))
                
                log_file.write(f"Addition completed:\n")
                log_file.write(f"  Added puzzles: {len(new_puzzles)}\n")
                log_file.write(f"  Total active puzzles: {len(trainer.active_puzzles)}\n")
                log_file.flush()
                
                trainer.advance_phase()
            
            # Plot progress after each phase
            if all_metrics_history:
                plot_phase_training_metrics(
                    all_metrics_history, 
                    all_accuracies_history,
                    trainer.get_phase_status(),
                    title=f"Phase-Based Training (Global Phase {trainer.global_phase_count})"
                )
            
            # Show debug info periodically
            if current_phase == "training" and len(trainer.active_puzzles) > 0:
                print(f"\n--- Phase Debug Info ---")
                puzzle = trainer.active_puzzles[0]
                puzzle_tensor = torch.tensor(
                    puzzle.test_input, 
                    dtype=torch.long, 
                    device=device
                ).unsqueeze(0)
                print_selection_debug(puzzle_tensor, sender, receiver, trainer)
            
            # Safety check to prevent infinite loops
            if trainer.global_phase_count >= max_global_phases:
                break
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL PHASE-BASED TRAINING SUMMARY")
    print("="*60)
    
    final_phase_info = trainer.get_phase_status()
    print(f"Completed global phases: {final_phase_info['global_phase_count']}")
    print(f"Final phase: {final_phase_info['current_phase']}")
    print(f"Final active puzzles: {final_phase_info['active_puzzles']}")
    print(f"Total removed symbols: {final_phase_info['removed_symbols']}")
    print(f"Repetitions per puzzle: {final_phase_info['selection_config']['repetitions_per_puzzle']}")
    print(f"Final puzzle-symbol mapping: {final_phase_info['puzzle_symbol_mapping']}")
    
    # Note if we stopped early
    if final_phase_info['current_phase'] in ['consolidation', 'addition']:
        print(f"Note: Stopped after final training phase, skipping {final_phase_info['current_phase']} phase")
    
    if all_metrics_history:
        recent_metrics = all_metrics_history[-50:]  # Last 50 steps
        final_acc1 = np.mean([m['agent1_selection_accuracy'] for m in recent_metrics])
        final_acc2 = np.mean([m['agent2_selection_accuracy'] for m in recent_metrics])
        print(f"\nFinal Performance:")
        print(f"  Agent 1 Selection Accuracy: {final_acc1:.3f}")
        print(f"  Agent 2 Selection Accuracy: {final_acc2:.3f}")
    
    sender.print_position_symbol_mapping()
    receiver.print_position_symbol_mapping()
    print("="*60)
    
    # Save final plot and cleanup live grapher
    if live_grapher:
        print("\nSaving final training plot...")
        live_grapher.save_final_plot("live_training_final.png")
        
        # Keep the plot window open for a bit to see final results
        print("Live training plot will remain open. Close the plot window to continue.")
        try:
            # Wait for user to close the plot window
            while plt.get_fignums():
                plt.pause(1.0)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            live_grapher.close()
    
    print("\nPhase-based training with puzzle repetitions complete! Check phase_training_log.txt for details")

if __name__ == "__main__":
    main()