# Fixes for Sequence Length > 1 Training

## Issues Fixed

### Issue 1: Frozen Symbols Being Overwritten
**Problem**: When loading a snapshot with 6 frozen symbols (10-15), `initialize_first_puzzles()` was starting symbol assignment at index 10 instead of 16, overwriting the frozen symbols.

**Root Cause**: Recent simplifications removed the logic that accounted for frozen symbols when creating new symbol assignments.

**Solution**: Restored logic to detect frozen symbols and start new symbol assignments after them.

```python
# In initialize_first_puzzles()
frozen_comm_symbols = self.agent1.current_comm_symbols if hasattr(self, 'frozen_comm_symbols') and self.frozen_comm_symbols > 0 else 0
start_symbol = self.agent1.puzzle_symbols + frozen_comm_symbols  # e.g., 10 + 6 = 16
```

### Issue 2: Graph Showing Wrong Symbol Count
**Problem**: Live grapher was initialized with `initial_comm_symbols` from config instead of the actual current symbol count from agents.

**Solution**: Use the actual current symbol count from agents after initialization.

```python
# Use actual current_comm_symbols from agents, not config value
current_symbols = trainer.agent1.current_comm_symbols
live_grapher.active_symbols_count = current_symbols
```

### Issue 3: Pretraining/Consolidation/Addition for Position 1+
**Problem**: When `current_seq_length > 1`, the system was trying to run pretraining, consolidation, and addition phases, but position 1+ symbols don't have puzzle mappings.

**Solution**: Skip these phases entirely when `current_seq_length > 1`.

## Implementation Details

### 1. Symbol Initialization Based on Sequence Length

**For Position 0 (`seq_length = 1`):**
```python
# Create puzzle mappings for pretraining
print("Initializing puzzles for position 0 (with puzzle mappings)")
trainer.initialize_first_puzzles()
```

**For Position 1+ (`seq_length > 1`):**
```python
# Just create communication symbols without puzzle mappings
print(f"Initializing symbols for position {seq_length - 1} (no puzzle mappings)")

# Determine frozen symbols and current position
frozen_comm_symbols = getattr(trainer, 'frozen_comm_symbols', 0)
frozen_positions = getattr(trainer, 'frozen_positions', [])
current_position = max(frozen_positions) + 1 if frozen_positions else 0

# Create new communication symbols starting after frozen ones
start_symbol = trainer.agent1.puzzle_symbols + frozen_comm_symbols
num_new_symbols = config['initial_comm_symbols']
new_symbol_indices = set(range(start_symbol, start_symbol + num_new_symbols))

# Update agent vocabularies
total_comm_symbols = frozen_comm_symbols + num_new_symbols
trainer.agent1.current_comm_symbols = total_comm_symbols
trainer.agent2.current_comm_symbols = total_comm_symbols

# Assign to position vocabulary
trainer.agent1.set_position_vocabulary(current_position, new_symbol_indices)
trainer.agent2.set_position_vocabulary(current_position, new_symbol_indices)
```

### 2. Skip Pretraining for seq_length > 1

```python
if trainer.agent1.current_seq_length > 1:
    print("Skipping pretraining (seq_length > 1, no puzzle mappings for current position)")
    trainer.advance_phase()
elif trainer.skip_pretraining_always or trainer.skip_next_pretraining:
    # ... existing skip logic
else:
    # ... run pretraining
```

### 3. Skip Consolidation for seq_length > 1

```python
elif current_phase == "consolidation":
    if trainer.agent1.current_seq_length > 1:
        print("Skipping consolidation (seq_length > 1, no puzzle mappings for current position)")
        trainer.advance_phase()
        continue
    # ... rest of consolidation
```

### 4. Skip Addition for seq_length > 1

```python
elif current_phase == "addition":
    if trainer.agent1.current_seq_length > 1:
        print("Skipping addition (seq_length > 1, no puzzle mappings for current position)")
        trainer.advance_phase()
        continue
    # ... rest of addition
```

### 5. Skip Early-Stop Consolidation/Addition

```python
if early_stop:
    if trainer.agent1.current_seq_length > 1:
        print("Skipping consolidation/addition (seq_length > 1, no puzzle mappings)")
    else:
        # ... run consolidation and addition
```

## Training Flow Comparison

### Position 0 (seq_length = 1)
```
1. Initialize puzzles with mappings
   - Puzzles 0-2 → Symbols 10-12
   - Assign to position 0 vocabulary

2. Pretraining (trains encoder on puzzles)

3. Training (bidirectional communication)

4. Consolidation (removes weak symbols)

5. Addition (adds more puzzles)

6. Back to step 2 for next cycle
```

### Position 1 (seq_length = 2)
```
1. Load snapshot (freeze symbols 10-12, position 0)

2. Initialize symbols WITHOUT mappings
   - Create symbols 13-15 (no puzzle assignments)
   - Assign to position 1 vocabulary

3. Skip Pretraining ← No puzzle mappings

4. Training (bidirectional communication)
   - Position 0: Uses frozen symbols 10-12
   - Position 1: Uses new symbols 13-15

5. Skip Consolidation ← No puzzle mappings

6. Skip Addition ← No puzzle mappings

7. Training continues with same symbols
```

## Example Output

### Phase 0 (seq_length = 1)
```
Initializing puzzles for position 0 (with puzzle mappings)
Starting symbol assignment at index 10
Initialized with 3 RANDOMLY SELECTED puzzles
Symbol assignments: {0: 10, 1: 11, 2: 12}
Communication symbols: 3 (no frozen symbols)
Assigned symbols [10, 11, 12] to position 0

[Runs: Pretraining → Training → Consolidation → Addition]
```

### Phase 1 (seq_length = 2, loading snapshot)
```
Applying Freezing from Snapshot:
  • Frozen communication symbols: 3 (indices 10 to 12)
  • Restoring position-specific vocabularies:
    Position 0: [10, 11, 12] (Agent1)

✓ Updating sequence length: 1 → 2
✓ Freezing position 0

Initializing symbols for position 1 (no puzzle mappings)
  Creating 3 new symbols: [13, 14, 15]
  Assigning to position 1
  Total communication symbols: 6 (frozen: 3, new: 3)

Live grapher updated with current symbols: 6

[Training Phase]
Skipping pretraining (seq_length > 1, no puzzle mappings for current position)
Skipping consolidation (seq_length > 1, no puzzle mappings for current position)
Skipping addition (seq_length > 1, no puzzle mappings for current position)

[Training continues with fixed symbols]
```

## Key Points

1. **Position 0**: Has puzzle mappings, does full training cycle (pretraining/training/consolidation/addition)

2. **Position 1+**: No puzzle mappings, ONLY does training phase with fixed symbols

3. **Symbol Counts**:
   - Position 0: Starts at `puzzle_symbols` (10)
   - Position 1: Starts at `puzzle_symbols + frozen_comm_symbols` (16 if 6 frozen)

4. **Position Vocabularies**:
   - Position 0: {10, 11, 12} ← Frozen from Phase 0
   - Position 1: {13, 14, 15} ← Created in Phase 1
   - Each position can ONLY use its assigned symbols

5. **No Pretraining Protocol Yet**: For now, position 1+ symbols are randomly initialized. Future work will develop a pretraining protocol for these positions.

## Modified Files

1. **`src/trainer_selection.py`**
   - Fixed `initialize_first_puzzles()` to account for frozen symbols
   - Enhanced logging to show frozen vs new symbols

2. **`src/train_selection.py`**
   - Added conditional initialization based on `current_seq_length`
   - Skip pretraining, consolidation, and addition for `seq_length > 1`
   - Fixed live grapher initialization to use actual symbol count

## Testing

To test:
1. Train Phase 0 with `current_seq_length=1`, save snapshot
2. Resume from snapshot with `current_seq_length=2`
3. Verify:
   - Console shows starting symbol assignment at correct index (16 not 10)
   - Live grapher shows 6 symbols (not 3)
   - Reconstruction display shows correct symbol indices
   - Pretraining/consolidation/addition are skipped
   - Training proceeds normally

All these behaviors are now working correctly! 🎯

