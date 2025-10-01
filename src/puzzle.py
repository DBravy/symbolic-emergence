import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union, Set
from dataclasses import dataclass

class Puzzle:
    def __init__(self, train_inputs: List[np.ndarray], 
                 train_outputs: List[np.ndarray],
                 test_input: np.ndarray):  # Changed from test_inputs
        """Initialize a puzzle with training examples and test input"""
        if len(train_inputs) == 0 or len(train_outputs) == 0:
            raise ValueError("Must provide at least one training example")
            
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_input = test_input  # Single test input
        self.test_inputs = [test_input]  # Keep list version for compatibility
        self.test_outputs = None
        
        # Validate all grids
        for grid in train_inputs + train_outputs + [test_input]:
            self.validate_grid(grid)
    
    @property
    def num_training_examples(self) -> int:
        return len(self.train_inputs)
    
    @property
    def grid_shape(self) -> tuple:
        return self.train_inputs[0].shape
    
    @property 
    def unique_colors(self) -> Set[int]:
        colors = set()
        for grid in self.train_inputs + self.train_outputs + [self.test_input]:
            colors.update(set(grid.ravel()))
        return colors
        
    @staticmethod
    def validate_grid(grid: np.ndarray) -> bool:
        """Check if grid has valid format and values"""
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid must be a numpy array")
        if grid.min() < 0 or grid.max() > 9:
            raise ValueError("Grid values must be between 0 and 9")
        return True
    
    @classmethod
    def from_single_example(cls, input_grid: np.ndarray, output_grid: np.ndarray):
        """Create puzzle from single training example (input-output pair)"""
        puzzle = cls(
            train_inputs=[input_grid],
            train_outputs=[output_grid],
            test_input=input_grid.copy()
        )
        # Store the output grid as test_output for input-output training
        puzzle.test_output = output_grid.copy()
        puzzle.test_outputs = [output_grid.copy()]
        return puzzle
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]):
        """Create puzzle from dictionary format"""
        train_inputs = [np.array(ex['input']) for ex in data['train']]
        train_outputs = [np.array(ex['output']) for ex in data['train']]
        test_input = np.array(data['test'][0]['input'])
        
        puzzle = cls(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_input=test_input
        )
        
        # Handle test outputs if available
        if all('output' in ex for ex in data['test']):
            puzzle.test_outputs = [np.array(ex['output']) for ex in data['test']]
        
        return puzzle
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert puzzle to dictionary format"""
        result = {
            'train': [
                {'input': inp.tolist(), 'output': out.tolist()}
                for inp, out in zip(self.train_inputs, self.train_outputs)
            ],
            'test': [{'input': self.test_input.tolist()}]
        }
        
        if self.test_outputs:
            result['test'][0]['output'] = self.test_outputs[0].tolist()
                
        return result
    
    def rotate90(self):
        """Return new puzzle with all grids rotated 90 degrees clockwise"""
        return type(self)(
            train_inputs=[np.rot90(grid, k=-1) for grid in self.train_inputs],
            train_outputs=[np.rot90(grid, k=-1) for grid in self.train_outputs],
            test_input=np.rot90(self.test_input, k=-1)
        )
        
    def transpose(self):
        """Return new puzzle with all grids transposed"""
        return type(self)(
            train_inputs=[grid.T for grid in self.train_inputs],
            train_outputs=[grid.T for grid in self.train_outputs],
            test_input=self.test_input.T
        )
    
    def permute_colors(self, permutation: List[int]):
        """Return new puzzle with color permutation applied to all grids"""
        def permute_grid(grid):
            grid_copy = grid.copy()
            for i, j in enumerate(permutation):
                grid_copy[grid == i] = j
            return grid_copy
            
        return type(self)(
            train_inputs=[permute_grid(grid) for grid in self.train_inputs],
            train_outputs=[permute_grid(grid) for grid in self.train_outputs],
            test_input=permute_grid(self.test_input)
        )

