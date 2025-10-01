#!/bin/bash

# Test script for single-puzzle ARC training

echo "============================================"
echo "Testing Single-Puzzle ARC Training"
echo "============================================"
echo ""

# Check if required files exist
echo "Checking required files..."

if [ ! -f "arc-agi_test_challenges.json" ]; then
    echo "ERROR: arc-agi_test_challenges.json not found"
    echo "Please ensure the ARC dataset is in the root directory"
    exit 1
fi
echo "✓ ARC dataset found"

if [ ! -f "src/train_single_puzzle.py" ]; then
    echo "ERROR: src/train_single_puzzle.py not found"
    exit 1
fi
echo "✓ Training script found"

if [ ! -f "web_app.py" ]; then
    echo "ERROR: web_app.py not found"
    exit 1
fi
echo "✓ Web app found"

if [ ! -f "templates/arc_single_puzzle.html" ]; then
    echo "ERROR: templates/arc_single_puzzle.html not found"
    exit 1
fi
echo "✓ Web template found"

echo ""
echo "============================================"
echo "Test 1: List available ARC puzzles"
echo "============================================"

# Get first few puzzle IDs
python3 << 'PYTHON'
import json
with open('arc-agi_test_challenges.json', 'r') as f:
    data = json.load(f)
puzzles = list(data.keys())[:5]
print(f"Found {len(data)} total puzzles")
print(f"First 5 puzzle IDs: {', '.join(puzzles)}")
PYTHON

echo ""
echo "============================================"
echo "Test 2: Load a puzzle and show structure"
echo "============================================"

python3 << 'PYTHON'
import json
with open('arc-agi_test_challenges.json', 'r') as f:
    data = json.load(f)
puzzle_id = list(data.keys())[0]
puzzle = data[puzzle_id]
print(f"Puzzle ID: {puzzle_id}")
print(f"  Training examples: {len(puzzle['train'])}")
for i, ex in enumerate(puzzle['train']):
    print(f"    Example {i+1}: input {len(ex['input'])}×{len(ex['input'][0])} -> output {len(ex['output'])}×{len(ex['output'][0])}")
print(f"  Test examples: {len(puzzle['test'])}")
PYTHON

echo ""
echo "============================================"
echo "Test 3: Verify Python dependencies"
echo "============================================"

python3 << 'PYTHON'
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")

try:
    import flask
    print(f"✓ Flask {flask.__version__}")
except ImportError:
    print("✗ Flask not installed")
PYTHON

echo ""
echo "============================================"
echo "Test 4: Quick training test (10 cycles)"
echo "============================================"

# Get first puzzle ID
PUZZLE_ID=$(python3 -c "import json; data = json.load(open('arc-agi_test_challenges.json')); print(list(data.keys())[0])")

echo "Running short training on puzzle: $PUZZLE_ID"
echo "Command: python src/train_single_puzzle.py --puzzle-id $PUZZLE_ID --cycles 10"
echo ""

python src/train_single_puzzle.py --puzzle-id "$PUZZLE_ID" --cycles 10

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
else
    echo ""
    echo "✗ Training failed"
    exit 1
fi

echo ""
echo "============================================"
echo "All tests passed!"
echo "============================================"
echo ""
echo "To use the web interface:"
echo "  1. Start the web server: python web_app.py"
echo "  2. Open browser: http://localhost:5001/arc-single-puzzle"
echo "  3. Select a puzzle and start training"
echo ""

