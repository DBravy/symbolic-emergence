# Addition Phase for Position 1+ (seq_length > 1)

## Issues Fixed

### Issue 1: Training Not Starting When Loading Snapshot
**Problem**: When `seq_length > 1`, we weren't initializing `active_puzzles`, causing training to fail.

**Solution**: Load puzzles for training (without creating puzzle-symbol mappings):
```python
# Select random puzzles for training (but don't map them to symbols)
available_indices = list(range(len(all_puzzles)))
selected_indices = random.sample(available_indices, config['initial_puzzle_count'])
trainer.active_puzzles = [all_puzzles[i] for i in selected_indices]

# Don't create puzzle_symbol_mapping - position 1+ symbols are not tied to puzzles
```

### Issue 2: No Addition of Symbols for Position 1+
**Problem**: Skipping addition entirely meant no new symbols were added for position 1, so the vocabulary never grew.

**Solution**: Addition adds NEW SYMBOLS (not puzzles) for position 1+:
```python
if trainer.agent1.current_seq_length > 1:
    # Add symbols without puzzle mappings
    start_symbol = trainer.agent1.current_total_symbols
    num_new_symbols = config['puzzles_per_addition']
    new_symbol_indices = set(range(start_symbol, start_symbol + num_new_symbols))
    
    # Update vocabularies
    trainer.agent1.current_comm_symbols += num_new_symbols
    trainer.agent1.current_total_symbols += num_new_symbols
    
    # Add to position vocabulary
    trainer.agent1.set_position_vocabulary(current_position, existing_vocab | new_symbol_indices)
else:
    # Position 0: Add puzzles with mappings (original behavior)
    new_puzzles = run_addition_phase(trainer)
```

## Training Flow Comparison

### Position 0 (seq_length = 1)
```
1. Initialize puzzles with mappings
   - Puzzles 0-2 → Symbols 10-12
   - puzzle_symbol_mapping = {0: 10, 1: 11, 2: 12}

2. Pretraining (trains encoder on puzzles)

3. Training (bidirectional communication)

4. Consolidation (removes weak symbols)

5. Addition
   - Add 3 more puzzles → Symbols 13-15
   - puzzle_symbol_mapping = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

6. Repeat from step 2
```

### Position 1 (seq_length = 2)
```
1. Load snapshot
   - Freeze symbols 10-15 (position 0)
   - Freeze position 0

2. Initialize for position 1
   - Load puzzles (for training context)
   - Create symbols 16-18 WITHOUT mappings
   - Position 1 vocabulary: {16, 17, 18}
   - puzzle_symbol_mapping = {} (empty!)

3. Skip Pretraining (no mappings)

4. Training (bidirectional communication)
   - Position 0: Uses frozen symbols 10-15
   - Position 1: Uses new symbols 16-18

5. Skip Consolidation (no mappings)

6. Addition
   - Add 3 more SYMBOLS (not puzzles) → 19-21
   - Position 1 vocabulary: {16, 17, 18, 19, 20, 21}
   - puzzle_symbol_mapping still empty

7. Training continues with expanded vocabulary

8. Repeat from step 6 (skip pretraining/consolidation, add symbols)
```

## Key Differences

| Aspect | Position 0 | Position 1+ |
|--------|-----------|------------|
| **Puzzles** | Mapped to symbols | Used for training only |
| **puzzle_symbol_mapping** | Full mappings | Empty |
| **Pretraining** | Yes | No |
| **Consolidation** | Yes | No |
| **Addition** | Adds puzzles+symbols | Adds symbols only |
| **Symbol Growth** | Via puzzle additions | Direct symbol additions |

## Example Console Output

### Phase 0 (seq_length = 1)
```
Initializing puzzles for position 0 (with puzzle mappings)
Starting symbol assignment at index 10
Initialized with 3 RANDOMLY SELECTED puzzles
Selected puzzle indices: [42, 157, 289]
Symbol assignments: {0: 10, 1: 11, 2: 12}

[Pretraining → Training → Consolidation → Addition cycles]
```

### Phase 1 (seq_length = 2, from snapshot)
```
Loading snapshot with 6 frozen symbols...
Restoring position vocabularies:
  Position 0: [10, 11, 12, 13, 14, 15]

Initializing for position 1 (training without puzzle mappings)
  Selected 3 puzzles for training: [88, 234, 512]
  Creating 3 new symbols: [16, 17, 18]
  Assigning to position 1
  Total communication symbols: 9 (frozen: 6, new: 3)
  Note: No puzzle-symbol mappings for position 1 (training only)

[Training Phase]
Skipping pretraining (seq_length > 1, no puzzle mappings)
[Training cycle completes]

Skipping consolidation (seq_length > 1, no puzzle mappings)

[Addition Phase]
============================================================
ADDITION PHASE - Adding symbols for position 1
============================================================
Creating 3 new symbols: [19, 20, 21]
Assigning to position 1
Total communication symbols: 12
Position 1 vocabulary: [16, 17, 18, 19, 20, 21]

[Training resumes with expanded vocabulary]
```

## Implementation Details

### 1. Initialization (train_selection.py)

```python
if trainer.agent1.current_seq_length == 1:
    # Position 0: puzzles + mappings
    trainer.initialize_first_puzzles()
else:
    # Position 1+: puzzles only (no mappings)
    trainer.active_puzzles = [select random puzzles]
    # Create symbols without mappings
    # Assign to position vocabulary
```

### 2. Addition Phase (train_selection.py)

```python
if trainer.agent1.current_seq_length > 1:
    # Add symbols to current position
    start_symbol = trainer.agent1.current_total_symbols
    new_symbol_indices = set(range(start_symbol, start_symbol + num_new_symbols))
    
    trainer.agent1.current_comm_symbols += num_new_symbols
    trainer.agent1.set_position_vocabulary(position, vocab | new_symbol_indices)
else:
    # Original behavior: add puzzles with mappings
    new_puzzles = run_addition_phase(trainer)
```

### 3. Early Stop Addition

Same logic applied when early stop is triggered - adds symbols for position 1+.

## Why This Works

1. **Position 0 needs puzzle mappings** because:
   - Pretraining teaches encoder to map puzzles → symbols
   - Consolidation tests which puzzles trigger which symbols
   - Symbol meanings are grounded in specific puzzles

2. **Position 1+ doesn't need puzzle mappings** because:
   - No pretraining (symbols start random, learn through training)
   - No consolidation (no puzzle-symbol relationship to test)
   - Symbol meanings emerge through training interactions
   - Puzzles provide training context but aren't tied to specific symbols

3. **Addition still needed** to expand vocabulary:
   - More symbols = more expressiveness
   - Can represent finer distinctions
   - Agents learn which symbols are useful through training

## Symbol Count Evolution

### Position 0 Training
```
Initial:  3 symbols  (10-12)
After 1:  6 symbols  (10-15)
After 2:  9 symbols  (10-18)
After 3: 12 symbols  (10-21)
```

### Position 1 Training (starting from 6 frozen)
```
Initial:  9 symbols  (10-15 frozen, 16-18 new)
After 1: 12 symbols  (10-15 frozen, 16-21 new)
After 2: 15 symbols  (10-15 frozen, 16-24 new)
After 3: 18 symbols  (10-15 frozen, 16-27 new)
```

Position 1 vocabulary grows while position 0 remains frozen!

## Modified Files

1. **`src/train_selection.py`**
   - Fixed initialization to load puzzles for position 1+
   - Modified addition phase to add symbols (not puzzles) for position 1+
   - Updated early-stop addition to add symbols

## Testing

1. Train Phase 0 with `seq_length=1`, save snapshot
2. Resume with `seq_length=2`
3. Verify:
   - Training starts successfully
   - Active puzzles are loaded
   - Initial symbols created (16-18)
   - Addition adds more symbols (19-21)
   - Position 1 vocabulary grows
   - Training continues normally

All working correctly now! 🎯

