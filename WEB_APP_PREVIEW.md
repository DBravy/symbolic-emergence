# Web App Preview: Single Puzzle Reconstruction Training

## Visual Guide to the Updated Interface

### Page Layout

```
┌───────────────────────────────────────────────────────────────────────┐
│                   Single-Puzzle ARC Training                          │
│  [Home] [ARC Solving] [Single Puzzle*]                               │
└───────────────────────────────────────────────────────────────────────┘

┌─────────────────────── SELECT ARC PUZZLE ─────────────────────────────┐
│                                                                         │
│  Choose a puzzle to train agents to communicate about...               │
│                                                                         │
│  [Load Puzzles]  Training Cycles: [100▼]                              │
│                                                                         │
│  ┌─ Puzzle List ──────────────────────────────────────────────────┐   │
│  │  ┌─ 007bbfb7 ────────────────┐  ← Click to select              │   │
│  │  │ Training: 4 | Test: 1     │                                  │   │
│  │  └────────────────────────────┘                                 │   │
│  │  ┌─ 00d62c1b ────────────────┐                                  │   │
│  │  │ Training: 3 | Test: 1     │                                  │   │
│  │  └────────────────────────────┘                                 │   │
│  │  [... 98 more puzzles ...]                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────── PUZZLE PREVIEW ─────────────────────────────────┐
│                                                                         │
│  Training Examples:                                                    │
│  ┌─ Example 1 ──┐  ┌─ Example 2 ──┐  ┌─ Example 3 ──┐                │
│  │ Input:       │  │ Input:       │  │ Input:       │                │
│  │ [Grid 1]     │  │ [Grid 2]     │  │ [Grid 3]     │                │
│  │ Output:      │  │ Output:      │  │ Output:      │                │
│  │ [Grid 1']    │  │ [Grid 2']    │  │ [Grid 3']    │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
│                                                                         │
│  Test Example:                                                         │
│  ┌─ Test 1 ─────┐                                                     │
│  │ Input:       │                                                     │
│  │ [Test Grid]  │                                                     │
│  │ Expected Out:│                                                     │
│  │ [Target Grid]│                                                     │
│  └──────────────┘                                                     │
│                                                                         │
│  [Start Training]                                                      │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────── TRAINING STATUS ────────────────────────────────┐
│  Status: Training                                                      │
│  Message: Cycle 45/100                                                │
│  Progress: 45%                                                        │
│  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░]                       │
│                                                                         │
│  [Stop Training]                                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────── RECONSTRUCTION RESULTS ────────────────────────────┐
│  (Shows after training completes)                                      │
│                                                                         │
│  ┌─ Test Input ──┐  ┌─ Expected ───┐  ┌─ Reconstructed ┐             │
│  │               │  │               │  │                │             │
│  │  ██░░██       │  │  ░░██░░       │  │  ░███░░        │             │
│  │  ░░██░░       │  │  ██░░██       │  │  ██░░██        │             │
│  │  ██░░██       │  │  ░░██░░       │  │  ░███░░        │             │
│  │               │  │               │  │                │             │
│  │               │  │               │  │  Accuracy: 75% │             │
│  └───────────────┘  └───────────────┘  └────────────────┘             │
│                                                                         │
│  Colors: ARC Palette (Black, Blue, Red, Green, Yellow, etc.)          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── TRAINING LOGS ────────────────────────────────┐
│  [2024-10-01 10:15:23] Training started on puzzle 007bbfb7            │
│  [2024-10-01 10:15:24] Using reconstruction training mode             │
│  [2024-10-01 10:15:25] Cycle   10: Loss=2.1234, Recon_Acc=0.234      │
│  [2024-10-01 10:15:26] Cycle   20: Loss=1.5678, Recon_Acc=0.456      │
│  [2024-10-01 10:15:27] Cycle   30: Loss=1.2345, Recon_Acc=0.589      │
│  [... more logs ...]                                                   │
│  [2024-10-01 10:17:45] Reconstruction accuracy: 0.750                 │
│  [2024-10-01 10:17:45] Reconstruction result saved to status file     │
│  [2024-10-01 10:17:45] Training Complete!                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Color Palette

Grids use the standard ARC color scheme:
- Black (0): `#000000` ■
- Blue (1): `#0074D9` ■
- Red (2): `#FF4136` ■
- Green (3): `#2ECC40` ■
- Yellow (4): `#FFDC00` ■
- Grey (5): `#AAAAAA` ■
- Magenta (6): `#F012BE` ■
- Orange (7): `#FF851B` ■
- Sky (8): `#7FDBFF` ■
- Maroon (9): `#870C25` ■

## Interactive Features

1. **Puzzle Selection**: Click any puzzle card to preview it
2. **Adjustable Cycles**: Increase/decrease training duration
3. **Live Progress**: Real-time status updates every 2 seconds
4. **Auto-Refresh**: Logs scroll automatically as new entries arrive
5. **Visual Comparison**: Side-by-side grid comparison on completion

## What to Expect

### During Training (2-10 minutes):
- Progress bar advancing
- Loss decreasing
- Accuracy increasing
- Logs streaming

### After Training:
- Three grids displayed
- Accuracy percentage shown
- Visual pattern comparison
- Can see what agents learned

### Interpretation:
- **High accuracy (>80%)**: Agents learned the pattern well
- **Medium accuracy (50-80%)**: Partial pattern learning
- **Low accuracy (<50%)**: Complex pattern or needs more cycles

