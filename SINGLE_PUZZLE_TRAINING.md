# Single-Puzzle ARC Training

## Overview

This feature allows you to train two agents to communicate about a single ARC puzzle using only **one communication symbol**. The agents learn to abstract the underlying pattern by communicating example input-output pairs back and forth, then test their understanding on the test example.

## Key Features

- **One Symbol Communication**: Agents use only a single communication symbol (position 0, sequence length 1)
- **Pattern Abstraction**: Agents must learn to encode the transformation rule, not just memorize examples
- **Bidirectional Learning**: Both agents learn to encode and decode, reinforcing the pattern
- **Interactive Web UI**: Select puzzles visually and monitor training in real-time

## How It Works

1. **Puzzle Selection**: Choose an ARC puzzle from the dataset (typically 3-4 training examples + 1 test)
2. **Training**: 
   - Agents communicate about each training example pair
   - Agent 1 encodes the input grid → produces symbol
   - Agent 2 receives symbol → selects correct output from candidates (including distractors)
   - This process repeats bidirectionally for all training pairs
3. **Testing**: After training, agents attempt to communicate about the test example
4. **Evaluation**: Check if the learned pattern generalizes to the unseen test case

## Architecture

```
Input Grid (e.g., 10×10) 
    ↓
Encoder (Agent 1)
    ↓
Single Symbol (1 position, vocabulary size 1)
    ↓
Decoder (Agent 2)  
    ↓
Selection from Candidates (correct output + distractors)
```

### Configuration

- **Embedding dimension**: 256
- **Hidden dimension**: 512
- **Puzzle symbols**: 10 (0-9, standard ARC colors)
- **Communication symbols**: 1
- **Sequence length**: 1 (minimal communication)
- **Learning rate**: 1e-5
- **Distractors**: 2 (makes selection task non-trivial)

## Usage

### Via Web Interface

1. Start the web server:
   ```bash
   python web_app.py
   ```

2. Navigate to: `http://localhost:5001/arc-single-puzzle`

3. Click "Load Puzzles" to see all available ARC puzzles

4. Click on a puzzle to preview its training examples and test case

5. Set the number of training cycles (default: 100)

6. Click "Start Training" to begin

7. Monitor progress in real-time via the status display and logs

### Via Command Line

```bash
cd /Users/djbray/Desktop/symbolic-communication

# Train on a specific puzzle
python src/train_single_puzzle.py --puzzle-id 00d62c1b --cycles 200

# Options:
#   --puzzle-id: Required. The ARC puzzle ID to train on
#   --cycles: Number of training cycles (default: 100)
#   --arc-file: Path to ARC JSON file (default: arc-agi_test_challenges.json)
#   --output-dir: Where to save checkpoints (default: ./outputs)
#   --status-file: Status file for web mode (default: training_status.json)
#   --device: cuda or cpu (default: auto-detect)
```

## Example Output

```
============================================================
Single-Puzzle ARC Training
============================================================
Device: cpu
Puzzle ID: 00d62c1b
Training cycles: 100

Loaded puzzle: 00d62c1b
  Training examples: 3
  Test examples: 1

  - Training pair: input (7, 7) -> output (7, 7)
  - Training pair: input (7, 7) -> output (7, 7)
  - Training pair: input (7, 7) -> output (7, 7)
  - Test: input (7, 7) -> output (7, 7)

Created 3 training pairs

Architecture:
  Embedding dim: 256
  Hidden dim: 512
  Puzzle symbols: 10
  Communication symbols: 1 (single symbol)
  Sequence length: 1

Starting training loop...
Training on 3 example pairs from puzzle 00d62c1b

Cycle    0: Loss=2.3456, Acc=0.333
Cycle   10: Loss=1.8901, Acc=0.500
Cycle   20: Loss=1.2345, Acc=0.667
...
Cycle  100: Loss=0.4567, Acc=0.889

============================================================
Testing on test example...
============================================================
Agent 1 encoded test input to symbol: 0
Agent 2 selection probabilities: [0.78, 0.12, 0.10]
Selected candidate: 0 (0=correct, 1+=2 distractors)
Correct: True

Checkpoint saved: ./outputs/single_puzzle_00d62c1b_20251001_123456.pt

============================================================
Training Complete!
============================================================
```

## Understanding the Results

### Success Indicators

1. **High Training Accuracy** (>80%): Agents can communicate about training examples
2. **Correct Test Selection**: Agent 2 selects the correct output for the test input
3. **Confidence**: High probability on correct candidate (>0.7)

### What It Means

- **Memorization**: If training accuracy is high but test fails → agents memorized examples
- **Generalization**: If test succeeds → agents learned the transformation pattern
- **Pattern Abstraction**: Single symbol forces compression, encouraging rule learning

## Research Questions

This setup allows you to explore:

1. **Can agents abstract patterns with minimal communication?**
   - One symbol vs. multiple symbols
   - Impact of sequence length

2. **How many training cycles are needed?**
   - Simple patterns vs. complex patterns
   - Relationship to number of examples

3. **Does the pattern type matter?**
   - Geometric transformations (rotation, reflection)
   - Color transformations
   - Object counting
   - Spatial relationships

4. **Role of distractors**
   - Impact on learning difficulty
   - Type of distractors (random vs. similar)

## Troubleshooting

### Training fails immediately
- Check that `arc-agi_test_challenges.json` exists in the root directory
- Verify puzzle ID is valid (use web UI to browse)
- Check Python dependencies are installed

### Low accuracy
- Try more training cycles (200-500)
- Some puzzles are very difficult
- Consider puzzle complexity vs. single symbol constraint

### Out of memory
- Reduce `embedding_dim` and `hidden_dim` in the script
- Use CPU instead of GPU for small models
- Process smaller grids

## Files Created

- `src/train_single_puzzle.py` - Training script
- `templates/arc_single_puzzle.html` - Web UI
- API endpoints in `web_app.py`:
  - `/api/arc/puzzles` - List all puzzles
  - `/api/arc/puzzle/<id>` - Get puzzle details
  - `/api/arc/train` - Start training

## Next Steps

1. Run training on multiple puzzles to find patterns
2. Compare single-symbol vs. multi-symbol communication
3. Analyze which puzzle types work best with minimal communication
4. Explore curriculum learning (easy → hard puzzles)
5. Investigate attention mechanisms to see what agents focus on

