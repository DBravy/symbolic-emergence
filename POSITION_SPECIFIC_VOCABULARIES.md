# Position-Specific Vocabularies for Progressive Sequence Training

## Overview

Implemented position-specific vocabulary constraints so that symbols learned in each training phase are permanently bound to specific positions in the message sequence.

## The Problem

Previously, all communication symbols could be used at any position in the sequence. This meant:
- Phase 0 symbols (10-12) could appear at any position
- Phase 1 symbols (13-14) could also appear at any position
- No enforcement of position-symbol relationships

## The Solution

### Core Concept

**Each training phase's symbols are locked to a specific position:**

- **Phase 0** (seq_length=1): Train symbols 10-12 → **Position 0 ONLY**
- **Phase 1** (seq_length=2): Train symbols 13-14 → **Position 1 ONLY**
- **Phase 2** (seq_length=3): Train symbols 15-16 → **Position 2 ONLY**
- **Phase N**: New symbols → **Position N ONLY**

### Implementation Details

#### 1. Agent-Level Position Vocabularies

**Added to `ProgressiveSelectionAgent` (`src/agent_selection.py`):**

```python
# Maps position index to set of allowed symbol indices
# Example: {0: {10, 11, 12}, 1: {13, 14}, 2: {15, 16}}
self.position_vocabularies = {}
```

**New Methods:**
- `set_position_vocabulary(position, symbol_indices)` - Assign symbols to a position
- `get_position_vocabularies()` - Get all position vocabularies
- `clear_position_vocabularies()` - Clear all restrictions

#### 2. Modified `get_position_symbol_mask()`

The encoder uses this method to determine which symbols are allowed at each position:

```python
def get_position_symbol_mask(self, position: int) -> torch.Tensor:
    """
    Get a mask for which symbols are allowed at a given sequence position.
    Uses position-specific vocabularies if set, otherwise allows all communication symbols.
    """
    device = next(self.parameters()).device
    mask = torch.zeros(self.max_num_symbols, dtype=torch.bool, device=device)
    
    if self.position_vocabularies and position in self.position_vocabularies:
        # Use position-specific vocabulary
        allowed_symbols = self.position_vocabularies[position]
        for sym_idx in allowed_symbols:
            if sym_idx < self.max_num_symbols:
                mask[sym_idx] = True
    else:
        # Fallback: allow all communication symbols
        mask[self.puzzle_symbols:self.current_total_symbols] = True
    
    return mask
```

#### 3. Automatic Position Assignment During Training

**In `initialize_first_puzzles()` (`src/trainer_selection.py`):**

```python
# Determine which position we're training
frozen_positions = getattr(self, 'frozen_positions', [])
if frozen_positions:
    current_training_position = max(frozen_positions) + 1
else:
    current_training_position = 0

# Get the symbol indices we just created
new_symbol_indices = set(self.puzzle_symbol_mapping.values())

# Assign these symbols ONLY to the current training position
self.agent1.set_position_vocabulary(current_training_position, new_symbol_indices)
self.agent2.set_position_vocabulary(current_training_position, new_symbol_indices)

print(f"Assigned symbols {sorted(new_symbol_indices)} to position {current_training_position}")
```

**In `add_new_puzzles()` (`src/trainer_selection.py`):**

Similar logic adds new symbols to the current unfrozen position's vocabulary.

#### 4. Snapshot Persistence

**Saving (`save_snapshot()`):**
```python
'trainer_state': {
    # ...
    'position_vocabularies_a1': {int(k): list(v) for k, v in self.agent1.get_position_vocabularies().items()},
    'position_vocabularies_a2': {int(k): list(v) for k, v in self.agent2.get_position_vocabularies().items()},
    # ...
}
```

**Loading (`train_selection.py`):**
```python
# Restore position vocabularies from snapshot
position_vocabs_a1 = trainer_state.get('position_vocabularies_a1', {})
position_vocabs_a2 = trainer_state.get('position_vocabularies_a2', {})

if position_vocabs_a1 or position_vocabs_a2:
    print(f"  • Restoring position-specific vocabularies:")
    for pos_str, symbols_list in position_vocabs_a1.items():
        pos = int(pos_str)
        symbols_set = set(symbols_list)
        trainer.agent1.set_position_vocabulary(pos, symbols_set)
        print(f"    Position {pos}: {sorted(symbols_set)} (Agent1)")
```

## Progressive Training Workflow

### Phase 0: Initial Training (seq_length=1)

```
Configuration:
  max_seq_length: 10      # Architecture capacity
  current_seq_length: 1   # Active positions

Initialize puzzles:
  Puzzles 0-2 → Symbols 10-12
  Assign to position 0: {10, 11, 12}

Training:
  Position 0: Can use ONLY {10, 11, 12}
  
Save snapshot:
  position_vocabularies_a1: {0: [10, 11, 12]}
  frozen_comm_symbols: 0
  frozen_positions: []
```

### Phase 1: Resume with Longer Sequence (seq_length=2)

```
Configuration:
  max_seq_length: 10      # Same
  current_seq_length: 2   # Increased!

Load snapshot:
  Restore position vocabularies: {0: {10, 11, 12}}
  Freeze symbols 10-12
  Freeze position 0

Initialize NEW puzzles:
  Puzzles 3-4 → Symbols 13-14
  Assign to position 1: {13, 14}

Training:
  Position 0: Can use ONLY {10, 11, 12} ← Frozen from Phase 0
  Position 1: Can use ONLY {13, 14}    ← New symbols for this phase
  
Agents learn to combine:
  - Position 0: Provides context from Phase 0 symbols
  - Position 1: Adds new information using Phase 1 symbols
  
Save snapshot:
  position_vocabularies_a1: {0: [10, 11, 12], 1: [13, 14]}
  frozen_comm_symbols: 3  # 10-12 from Phase 0
  frozen_positions: [0]
```

### Phase 2: Even Longer Sequence (seq_length=3)

```
Configuration:
  max_seq_length: 10
  current_seq_length: 3

Load snapshot:
  Restore position vocabularies: {0: {10, 11, 12}, 1: {13, 14}}
  Freeze symbols 10-14 (all previous phases)
  Freeze positions 0-1

Initialize NEW puzzles:
  Puzzles 5-6 → Symbols 15-16
  Assign to position 2: {15, 16}

Training:
  Position 0: {10, 11, 12} ← Frozen
  Position 1: {13, 14}     ← Frozen
  Position 2: {15, 16}     ← New!
  
Save snapshot:
  position_vocabularies_a1: {0: [10, 11, 12], 1: [13, 14], 2: [15, 16]}
  frozen_comm_symbols: 5   # 10-14 from previous phases
  frozen_positions: [0, 1]
```

## Example Training Output

### Phase 0 (Fresh Start)
```
Initialized with 3 RANDOMLY SELECTED puzzles
Selected puzzle indices: [42, 157, 289]
Symbol assignments: {0: 10, 1: 11, 2: 12}
Assigned symbols [10, 11, 12] to position 0

[Agent1] Position-Symbol Mapping (Phase-Based Selection):
  Similarity Metric: cosine
  Total communication symbols: 3
  Current sequence length: 1
  Position-specific vocabularies (Progressive Training):
    Position 0: symbols [10, 11, 12]
```

### Phase 1 (Resume from Snapshot)
```
Loading Weights and Applying Freezing
────────────────────────────────────────────────────────────

Applying Freezing from Snapshot:
────────────────────────────────────────────────────────────
  • No frozen positions (first phase)
  • Frozen communication symbols: 3
    (indices 10 to 12)
  • Setting agent vocabulary to start AFTER frozen symbols
  • Agents now have 3 symbols (frozen, will not be modified)
  • Next available symbol will be: 13
  • Fresh puzzles will be initialized and assigned to NEW symbols
  • Restoring position-specific vocabularies:
    Position 0: [10, 11, 12] (Agent1)
────────────────────────────────────────────────────────────

✓ Updating sequence length: 1 → 2
✓ Freezing position 0 (learned in previous phase)

Initialized with 2 RANDOMLY SELECTED puzzles
Selected puzzle indices: [88, 234]
Symbol assignments: {0: 13, 1: 14}
Assigned symbols [13, 14] to position 1

[Agent1] Position-Symbol Mapping (Phase-Based Selection):
  Similarity Metric: cosine
  Total communication symbols: 5
  Current sequence length: 2
  Position-specific vocabularies (Progressive Training):
    Position 0: symbols [10, 11, 12]  ← From Phase 0
    Position 1: symbols [13, 14]      ← New in Phase 1
```

## Key Benefits

### 1. **True Progressive Learning**
Each phase builds on stable, unchanging foundations from previous phases:
- Position 0 symbols never change their meaning across phases
- Position 1 symbols are consistent once learned
- No interference between positions

### 2. **Compositional Structure**
Messages have a clear structure where each position has a specific role:
- Earlier positions: Coarse-grained or high-level information
- Later positions: Fine-grained refinements or details

### 3. **Simplified Learning Problem**
Each phase only needs to learn:
- How to use NEW symbols at the NEW position
- How the new position interacts with frozen positions

### 4. **Better Generalization**
Position-specific vocabularies force structured communication:
- Can't "cheat" by putting any symbol at any position
- Must develop meaningful positional semantics

### 5. **Debugging & Interpretability**
Easy to understand what each position contributes:
- Position 0: First-level categorization
- Position 1: Second-level refinement
- Position 2: Third-level detail

## Implementation Files

### Modified Files
1. **`src/agent_selection.py`**
   - Added `position_vocabularies` tracking
   - Added `set_position_vocabulary()`, `get_position_vocabularies()`, `clear_position_vocabularies()`
   - Modified `get_position_symbol_mask()` to use position vocabularies
   - Updated `print_position_symbol_mapping()` to show position vocabularies

2. **`src/trainer_selection.py`**
   - Modified `initialize_first_puzzles()` to assign symbols to positions
   - Modified `add_new_puzzles()` to assign new symbols to current position
   - Modified `save_snapshot()` to include position vocabularies

3. **`src/train_selection.py`**
   - Modified snapshot loading to restore position vocabularies
   - Prints position vocabulary information during startup

## Testing Workflow

### Quick Test

1. **Phase 0: Train with seq_length=1**
   ```bash
   # In web UI:
   max_seq_length: 10
   current_seq_length: 1
   initial_comm_symbols: 3
   ```
   - Run training
   - Save snapshot "phase0"

2. **Phase 1: Resume with seq_length=2**
   ```bash
   # In web UI:
   Resume from: phase0_TIMESTAMP.pt
   current_seq_length: 2
   initial_comm_symbols: 2
   ```
   - Check console output for position vocabularies
   - Position 0 should be frozen with Phase 0 symbols
   - Position 1 should have new symbols

3. **Verify Enforcement**
   - During training, encoder should only predict allowed symbols at each position
   - Reconstruction samples should show messages respecting position constraints

## Summary

Position-specific vocabularies transform progressive training from "add more symbols and positions" into "build a structured compositional language" where each position has a fixed, learned role. This creates a clear hierarchy of information and prevents symbol interference across positions! 🎯

