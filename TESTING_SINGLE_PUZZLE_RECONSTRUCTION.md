# Testing Single Puzzle Reconstruction Training

## Quick Test Guide

### 1. Start the Web App
```bash
cd /Users/djbray/Desktop/symbolic-communication
python web_app.py
```

The app will start on http://localhost:5001

### 2. Navigate to Single Puzzle Training
- Open browser to http://localhost:5001
- Click on "Single Puzzle" in the navigation
- Or go directly to http://localhost:5001/arc-single-puzzle

### 3. Select a Puzzle
- Click "Load Puzzles" button
- You should see a list of 100 ARC puzzles
- Click on any puzzle to select it (e.g., "007bbfb7")
- Preview will show the training and test examples

### 4. Start Training
- Set training cycles (default 100, or use 50 for faster testing)
- Click "Start Training"
- You should see:
  - Training status section appears
  - Progress updates every 2 seconds
  - Logs showing cycle progress and loss/accuracy

### 5. View Results
When training completes, you should see:
- Status changes to "Complete"
- "Reconstruction Results" section appears showing:
  - Test Input grid (in ARC colors)
  - Expected Output grid
  - Reconstructed Output grid
  - Accuracy percentage

### Expected Behavior

#### Good Signs:
- ✓ Reconstruction accuracy improves over cycles
- ✓ Final accuracy > 50% after 100 cycles
- ✓ Reconstructed grid shows similar patterns to expected output
- ✓ Grid colors match ARC palette (black, blue, red, green, etc.)

#### What to Look For:
- Early cycles: Random-looking reconstructions (low accuracy)
- Mid cycles: Some patterns emerging (30-60% accuracy)
- Late cycles: Clear patterns matching expected output (60-95% accuracy)

### Troubleshooting

#### "Training script not found"
- Ensure `src/train_single_puzzle.py` exists
- Check file permissions

#### "ARC dataset file not found"
- Ensure `arc-agi_test_challenges.json` exists in root directory
- Or in `src/` directory

#### Training starts but no reconstruction shown
- Check browser console for JavaScript errors
- Verify `training_status.json` file is created
- Check that file contains `reconstruction` field

#### Low accuracy after training
- This is normal for complex puzzles with only 100 cycles
- Try increasing cycles to 200-500
- Or try simpler puzzles

### Example Test Case

**Quick 50-cycle test on simple puzzle:**
```
1. Select puzzle: 007bbfb7
2. Set cycles: 50
3. Start training
4. Wait ~2-5 minutes (depending on hardware)
5. View reconstruction results
6. Expected: 30-60% accuracy (simple patterns visible)
```

**Full 100-cycle test:**
```
1. Select puzzle: 007bbfb7
2. Set cycles: 100
3. Start training
4. Wait ~5-10 minutes
5. View reconstruction results
6. Expected: 60-80% accuracy (clear patterns)
```

## Debugging Tips

### Check Status File
```bash
cat training_status.json
```

Should show:
```json
{
  "status": "complete",
  "reconstruction": {
    "test_input": [[0, 1, 2, ...], ...],
    "test_output": [[0, 1, 2, ...], ...],
    "reconstructed": [[0, 1, 2, ...], ...],
    "accuracy": 0.75
  }
}
```

### Check Logs
Look in the web interface logs section for:
- "Using reconstruction training mode"
- "Cycle XXXX: Loss=X.XXX, Recon_Acc=X.XXX"
- "Reconstruction accuracy: X.XXX"
- "Reconstruction result saved to status file"

### Manual Script Test (if needed)
```bash
cd /Users/djbray/Desktop/symbolic-communication
python src/train_single_puzzle.py --puzzle-id 007bbfb7 --cycles 50
```

