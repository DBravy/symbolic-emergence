# Reconstruction Training Flow

## Before (Selection Training)
```
Training:
  Input Grid → Agent1 Encoder → Message Symbol
  Message Symbol + [Candidates] → Agent2 Selector → Pick best candidate
  
Testing:
  Test Input → Agent1 Encoder → Message
  Message + [Test Output + Distractors] → Agent2 Selector → Selection
  
Output: Which candidate was selected (binary correct/incorrect)
```

## After (Reconstruction Training)
```
Training (Bidirectional):
  Direction 1:
    Input Grid → Agent1 Encoder → Message Symbol
    Message Symbol → Agent2 Decoder → Reconstructed Output Grid
    Loss = CrossEntropy(Reconstructed, Target Output)
  
  Direction 2:
    Input Grid → Agent2 Encoder → Message Symbol
    Message Symbol → Agent1 Decoder → Reconstructed Output Grid
    Loss = CrossEntropy(Reconstructed, Target Output)
    
Testing:
  Test Input → Agent1 Encoder → Message Symbol
  Message Symbol → Agent2 Decoder → Reconstructed Output Grid
  
Output: 
  - Full reconstructed grid (viewable in web interface)
  - Pixel-by-pixel accuracy vs expected output
  - Visual comparison of expected vs reconstructed
```

## Web Interface Display

### Training Phase
Shows:
- Status: "Training"
- Progress: X%
- Current cycle and loss/accuracy metrics

### Completion Phase
Shows three grids side-by-side:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Test Input     │  │ Expected Output │  │ Reconstructed   │
│                 │  │                 │  │     Output      │
│   [Grid View]   │  │   [Grid View]   │  │   [Grid View]   │
│                 │  │                 │  │                 │
│                 │  │                 │  │  Accuracy: XX%  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

Each grid uses the standard ARC color palette for easy visual comparison.

## Key Advantages

1. **Visual Feedback**: See exactly what the agents learned
2. **Interpretability**: Understand what patterns the agents communicate
3. **Debugging**: Identify if agents are learning or just memorizing
4. **Better Training**: Reconstruction is a more direct learning objective
5. **Gradual Learning**: Can see how reconstruction improves over cycles

