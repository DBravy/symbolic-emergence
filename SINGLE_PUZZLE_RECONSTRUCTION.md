# Single Puzzle Reconstruction Training

## Overview
Modified the single puzzle training feature to use **reconstruction training** instead of selection training, and added visualization of the reconstructed output in the web app.

## Changes Made

### 1. Training Script (`src/train_single_puzzle.py`)

#### Training Loop
- **Changed from**: Selection-based training (agents select from candidate grids)
- **Changed to**: Reconstruction training (agents encode inputs and decode to reconstruct outputs)

**Key modifications:**
- Agents now train bidirectionally:
  - Agent1 encodes input → Agent2 decodes to reconstruct output
  - Agent2 encodes input → Agent1 decodes to reconstruct output
- Loss computed using cross-entropy between reconstructed and target grids
- Metrics track reconstruction accuracy instead of selection accuracy

#### Testing/Evaluation
- After training completes, the test input is encoded and decoded
- The reconstructed output grid is saved to the status file
- Accuracy is computed by comparing reconstructed vs expected output
- Status file now includes:
  ```json
  {
    "status": "complete",
    "reconstruction": {
      "test_input": [...],
      "test_output": [...],
      "reconstructed": [...],
      "accuracy": 0.XX
    }
  }
  ```

### 2. Web Template (`templates/arc_single_puzzle.html`)

#### New Section
Added a "Reconstruction Results" section that displays:
- Test Input grid
- Expected Output grid
- Reconstructed Output grid
- Reconstruction accuracy percentage

#### JavaScript Updates
- Added `displayReconstruction()` function to render the three grids side by side
- Status polling automatically shows reconstruction results when training completes
- Uses the existing `renderGrid()` function for consistent ARC color palette

### 3. Web App (`web_app.py`)
- No changes needed! The existing status polling API already handles the updated status file format

## How It Works

1. **Training Phase**: 
   - Each training cycle processes all example pairs
   - For each pair, both agents learn to encode inputs and decode to outputs
   - Loss backpropagates through encoder → message → decoder pathway

2. **Testing Phase**:
   - Agent1 encodes the test input into a single communication symbol
   - Agent2 decodes that symbol to reconstruct the test output
   - Reconstruction is saved for visualization

3. **Visualization**:
   - Web interface polls status file
   - When complete, reconstruction grids are displayed
   - User can visually compare expected vs reconstructed outputs

## Usage

1. Navigate to "Single Puzzle" page in web app
2. Select an ARC puzzle from the list
3. Click "Start Training" (default 100 cycles)
4. Watch training progress
5. When complete, view the reconstruction results showing:
   - How well the agents learned to communicate about the puzzle
   - Visual comparison of expected vs reconstructed outputs

## Benefits

- **More interpretable**: You can see what the agents actually reconstructed
- **Better for learning**: Reconstruction is a more direct objective for learning communication
- **Visual feedback**: Easy to see if communication is working
- **Debugging**: Can identify if agents are learning patterns or just memorizing

