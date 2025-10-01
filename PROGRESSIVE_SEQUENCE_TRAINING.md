# Progressive Sequence Training with Position Freezing

## Overview

This system implements progressive sequence training where each phase learns a new position in the message sequence while freezing previous positions and their associated communication symbols.

## Architecture

### Key Concept
- **Architecture Capacity**: `max_seq_length=10` (fixed across all phases)
- **Active Positions**: `current_seq_length` (increases per phase)
- **Frozen Symbols**: Communication symbols learned in previous phases are frozen

### Training Phases

```
Phase 0: current_seq_length=1
  - Learn position 0 with N communication symbols
  - Save snapshot with N symbols
  
Phase 1: current_seq_length=2
  - Load snapshot
  - Position 0: FROZEN (uses N frozen symbols from phase 0)
  - Position 1: TRAINABLE (learns NEW symbols starting from index N+10)
  
Phase 2: current_seq_length=3
  - Load snapshot from phase 1
  - Positions 0-1: FROZEN
  - Position 2: TRAINABLE (learns new symbols)
```

## Implementation Details

### 1. Agent (`agent_selection.py`)

#### New Methods

**`freeze_positions(positions: List[int])`**
- Freezes position predictor networks for specified positions
- Makes `requires_grad = False` for all parameters in those predictors

**`freeze_communication_symbols(num_symbols: int)`**
- Freezes embedding weights for the first N communication symbols
- Uses gradient hooks to zero out gradients for frozen symbol indices
- Frozen symbols: indices `[puzzle_symbols : puzzle_symbols + num_symbols]`

**`get_frozen_positions() -> List[int]`**
- Returns list of currently frozen positions

**`get_frozen_communication_symbols() -> int`**
- Returns count of frozen communication symbols

### 2. Trainer (`trainer_selection.py`)

#### New State Variables
```python
self.frozen_positions = []        # Track frozen position indices
self.frozen_comm_symbols = 0      # Track count of frozen communication symbols
```

#### Snapshot Format
```python
{
    'agent1_state_dict': {...},
    'agent2_state_dict': {...},
    'architecture': {
        'embedding_dim': 512,
        'hidden_dim': 1024,
        'num_symbols': 100,
        'puzzle_symbols': 10,
        'max_seq_length': 10,      # Fixed architecture capacity
        'similarity_metric': 'cosine'
    },
    'trainer_state': {
        'current_seq_length': 1,           # Active during this phase
        'frozen_positions': [0],           # Positions frozen during this phase
        'frozen_comm_symbols': 9,          # Number of frozen symbols
        'current_comm_symbols_a1': 9,      # Total symbols active
        # ... other state ...
    },
    'meta': {
        'created_at': '...',
        'format_version': 1
    }
}
```

### 3. Training Script (`train_selection.py`)

#### New Command-Line Argument
```bash
--resume-from PATH    # Path to snapshot file for progressive training
```

#### Snapshot Loading Logic

1. **Load State Dictionaries**
   ```python
   trainer.agent1.load_state_dict(snapshot['agent1_state_dict'])
   trainer.agent2.load_state_dict(snapshot['agent2_state_dict'])
   ```

2. **Apply Position Freezing**
   ```python
   frozen_positions = snapshot['trainer_state']['frozen_positions']
   trainer.agent1.freeze_positions(frozen_positions)
   trainer.agent2.freeze_positions(frozen_positions)
   ```

3. **Apply Symbol Freezing**
   ```python
   frozen_comm_symbols = snapshot['trainer_state']['frozen_comm_symbols']
   trainer.agent1.freeze_communication_symbols(frozen_comm_symbols)
   trainer.agent2.freeze_communication_symbols(frozen_comm_symbols)
   ```

4. **Update Sequence Length** (if specified in config)
   ```python
   if config['current_seq_length'] > old_seq_length:
       # Expand to new length
       # Automatically freeze previous positions
   ```

### 4. Web UI

#### Configuration Panel (Architecture Tab)
```html
Max Sequence Length (Architecture): 10
  └─ Maximum positions the model can support (fixed across phases)

Current Sequence Length (Active): 1
  └─ Number of positions to train in this phase
```

#### API Endpoints
- **`/api/snapshot/inspect`** - Extract architecture and state from snapshot
  ```json
  POST /api/snapshot/inspect
  { "filename": "comm_snapshot_20250101_120000.pt" }
  
  Response:
  {
    "architecture": { "max_seq_length": 10, ... },
    "trainer_state": { "frozen_positions": [0], "frozen_comm_symbols": 9, ... },
    "meta": { ... }
  }
  ```

## Usage Workflow

### Phase 0: Initial Training (1-Symbol Messages)

```bash
# Via Web UI
1. Set Configuration:
   - max_seq_length: 10
   - current_seq_length: 1
   - initial_comm_symbols: 4

2. Start Training

3. Train through multiple global phases with:
   - Symbol addition
   - Symbol consolidation
   - End with ~9 communication symbols

4. Save Snapshot (button in UI)
   - Saves: outputs/snapshots/comm_snapshot_TIMESTAMP.pt
```

**Snapshot Contains:**
- Position 0 weights (trained)
- 9 communication symbol embeddings (trained)
- frozen_positions: [] (none frozen yet)
- frozen_comm_symbols: 0 (will be set to 9 on load)

### Phase 1: Expanding to 2-Symbol Messages

```bash
# Command Line
python src/train_selection.py \
    --config training_config.json \
    --web-mode \
    --resume-from ./outputs/snapshots/comm_snapshot_TIMESTAMP.pt

# Config Changes:
# - current_seq_length: 2  (expand from 1 to 2)
```

**What Happens on Load:**
1. Loads agent weights from phase 0
2. Reads that phase 0 had 9 communication symbols active
3. **Freezes position 0 predictor** (can't modify how position 0 works)
4. **Freezes symbols 10-18** (the 9 learned communication symbols)
5. Sets current_seq_length = 2

**During Training:**
- Position 0: Uses frozen symbols 10-18 (no gradient updates)
- Position 1: Learns NEW symbols starting from 19+ (trainable)
- Agents learn to combine position 0 (frozen, known meanings) with position 1 (new, learnable) to improve selection accuracy

### Phase 2: Expanding to 3-Symbol Messages

```bash
python src/train_selection.py \
    --config training_config.json \
    --web-mode \
    --resume-from ./outputs/snapshots/phase1_snapshot_TIMESTAMP.pt

# Config: current_seq_length: 3
```

**Progressive Freezing:**
- Positions 0-1: FROZEN
- Symbols from phases 0-1: FROZEN
- Position 2: TRAINABLE with new symbols

## Verification

### Check Frozen State
```python
# In Python/during training:
print(f"Frozen positions: {trainer.agent1.get_frozen_positions()}")
print(f"Frozen symbols: {trainer.agent1.get_frozen_communication_symbols()}")
```

### Verify Gradients
```python
# Position 0 should have no gradients:
for param in trainer.agent1.encoder.position_predictors[0].parameters():
    assert not param.requires_grad, "Position 0 should be frozen!"

# Frozen symbols should get zero gradients:
# (gradient hook automatically zeros them during backward pass)
```

### Monitor Training
- Frozen symbols won't change values across epochs
- Loss should initially be higher (learning to use new position)
- Accuracy should improve as position 1 learns complementary information

## Key Benefits

1. **Architectural Compatibility**: `max_seq_length` stays fixed, so snapshots are always compatible
2. **Incremental Learning**: Each phase builds on previous learning without catastrophic forgetting
3. **Efficient Exploration**: Frozen positions provide stable foundation while new positions explore
4. **Controlled Complexity**: Add one position at a time, ensuring proper learning before expansion

## Notes

- Symbol indices: `[0-9: puzzle symbols] [10+: communication symbols]`
- Frozen communication symbols remain in embedding table but receive zero gradients
- New symbols can still be added via vocabulary expansion (trainer's consolidation/addition phases)
- Position predictors are separate networks, so freezing one doesn't affect others
- The system automatically freezes previous positions when expanding sequence length

