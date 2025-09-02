import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from puzzle import Puzzle

def test_valid_puzzle_creation():
    puzzle = Puzzle(
        train_inputs=[np.array([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7]])],
        train_outputs=[np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
        test_input=np.array([[0, 1], [2, 3]])
    )
    assert puzzle.num_training_examples == 2
    assert puzzle.validate_grid(puzzle.test_input)

def test_single_example_creation():
    input_grid = np.array([[0, 1], [2, 3]])
    output_grid = np.array([[1, 2], [3, 4]])
    puzzle = Puzzle.from_single_example(input_grid, output_grid)
    assert puzzle.num_training_examples == 1
    assert puzzle.validate_grid(puzzle.test_input)

def test_invalid_color_values():
    with pytest.raises(ValueError):
        Puzzle(
            train_inputs=[np.array([[0, 10], [2, 3]])],  # 10 is invalid
            train_outputs=[np.array([[1, 2], [3, 4]])],
            test_input=np.array([[0, 1], [2, 3]])
        )

def test_puzzle_transformations():
    puzzle = Puzzle(
        train_inputs=[np.array([[0, 1], [2, 3]])],
        train_outputs=[np.array([[1, 2], [3, 4]])],
        test_input=np.array([[0, 1], [2, 3]])
    )
    
    # Test rotation (clockwise)
    rotated = puzzle.rotate90()
    np.testing.assert_array_equal(
        rotated.train_inputs[0],
        np.array([[2, 0], [3, 1]])
    )
    
    # Test transpose
    transposed = puzzle.transpose()
    np.testing.assert_array_equal(
        transposed.train_inputs[0],
        np.array([[0, 2], [1, 3]])
    )
    
    # Test color permutation
    permuted = puzzle.permute_colors([1,0,3,2,4,5,6,7,8,9])
    np.testing.assert_array_equal(
        permuted.train_inputs[0],
        np.array([[1, 0], [3, 2]])
    )

def test_dict_conversion():
    data = {
        'train': [
            {'input': [[0, 1], [2, 3]], 'output': [[1, 2], [3, 4]]},
            {'input': [[4, 5], [6, 7]], 'output': [[5, 6], [7, 8]]}
        ],
        'test': [{'input': [[0, 1], [2, 3]]}]
    }
    
    puzzle = Puzzle.from_dict(data)
    assert puzzle.num_training_examples == 2
    
    converted = puzzle.to_dict()
    assert converted == data