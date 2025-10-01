# Frozen Symbols Protection

## Problem

When resuming from a snapshot with frozen communication symbols, the system was:
1. ✅ Correctly freezing the symbol embeddings
2. ❌ **BUT** then reassigning those symbols to new puzzles
3. ❌ Attempting to train frozen symbols in pretraining
4. ❌ Testing frozen symbols in consolidation (potentially removing them)

**Example**: Snapshot had 3 symbols (10-12) learned in Phase 0. When starting Phase 1:
- Symbols 10-12 were frozen (correct)
- But then `initialize_first_puzzles()` assigned 3 NEW puzzles to symbols 10-12
- Pretraining tried to train these frozen symbols
- Consolidation could potentially remove them if accuracy was low

## Solution

Implemented comprehensive protection for frozen symbols across all training phases:

### 1. Initialize Fresh Puzzles (Don't Restore Old Mappings)

When loading a snapshot, we **don't** restore old puzzle-symbol mappings. Instead:

```python
# Always initialize puzzles, even when loading from snapshot
# Frozen symbols won't be used for these puzzles
# New puzzles will get NEW symbols starting after frozen symbols
print("Initializing puzzles for this training phase")
trainer.initialize_first_puzzles()
```

**Result**: Fresh puzzles are assigned to NEW symbols (starting after frozen symbols).

**Why**: Frozen symbols are only used during training (in frozen positions). They don't need puzzle assignments for pretraining or consolidation since they're protected from both.

### 2. Update Agent Vocabulary After Freezing

```python
# Set agent vocabulary to include frozen symbols
trainer.agent1.current_comm_symbols = frozen_comm_symbols
trainer.agent1.current_total_symbols = puzzle_symbols + frozen_comm_symbols
trainer.agent1.communication_vocabulary = set(range(current_total_symbols))
```

**Result**: Agents know they have N frozen symbols at the start.

### 3. Skip Frozen Symbols in Pretraining

```python
# In run_pretraining_phase()
frozen_symbol_start = agent1.puzzle_symbols  # e.g., 10
frozen_symbol_end = frozen_symbol_start + trainer.frozen_comm_symbols  # e.g., 13

for puzzle in target_puzzles:
    symbol_idx = trainer.puzzle_symbol_mapping[active_idx]
    if frozen_symbol_start <= symbol_idx < frozen_symbol_end:
        # This puzzle is mapped to a frozen symbol - skip it
        skipped_frozen += 1
        continue
    # ... train on this puzzle
```

**Result**: Pretraining never touches puzzles mapped to frozen symbols (in practice, no puzzles will be mapped to frozen symbols initially).

### 4. Protect Frozen Symbols in Consolidation

```python
# In identify_recessive_symbols()
frozen_start = self.agent1.puzzle_symbols
frozen_end = frozen_start + self.frozen_comm_symbols

for symbol, predictions in confusion_data.items():
    # Skip frozen symbols - they should never be removed
    if frozen_start <= symbol < frozen_end:
        print(f"Symbol {symbol}: FROZEN (skipping consolidation analysis)")
        continue
    # ... analyze this symbol
```

**Result**: Frozen symbols are never considered for removal, regardless of accuracy.

## Console Output

When loading a snapshot with frozen symbols, you'll now see:

```
============================================================
Loading Weights and Applying Freezing
============================================================

✓ Loaded agent state dictionaries

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
────────────────────────────────────────────────────────────

✓ Successfully loaded snapshot and applied freezing
============================================================

Initializing puzzles for this training phase
Initialized with 3 RANDOMLY SELECTED puzzles
Selected puzzle indices: [100, 200, 300]
Symbol assignments: {0: 13, 1: 14, 2: 15}  ← NEW symbols!
Next available symbol: 16

============================================================
PRETRAINING PHASE - Encoder Training (100 epochs)
============================================================
Training on 3 puzzles with symbol mappings
Skipping 0 puzzles without mappings
[Pretraining trains on NEW puzzles with NEW symbols 13-15]
```

Notice: **Pretraining trains on NEW puzzles** with NEW symbols (13-15). Frozen symbols (10-12) are not used.

## Progressive Training Workflow

### Phase 0: Initial Training
```
Start:
  - No frozen symbols
  - Initialize 3 puzzles → symbols 10, 11, 12
  
Pretraining:
  - Trains on puzzles 0, 1, 2 (symbols 10-12)
  
Training:
  - Uses symbols 10-12 for sequence position 0
  
Consolidation:
  - Can remove symbols 10-12 if accuracy is low
  - Typically keeps them if trained well
  
Result:
  - End with 3 symbols (10-12) learned
  - Save snapshot
```

### Phase 1: Resume from Snapshot
```
Load Snapshot:
  - Freeze symbols 10-12
  - Set current_seq_length = 2
  - Freeze position 0
  - Set vocabulary to start at symbol 13
  
Initialize Fresh Puzzles:
  - Select NEW puzzles (not from Phase 0)
  - Assign to NEW symbols starting at 13
  
Pretraining:
  - Trains on NEW puzzles with symbols 13-15
  - Frozen symbols 10-12 are NOT used
  
Training:
  - Position 0: Uses frozen symbols 10-12
  - Position 1: Uses NEW symbols 13-15
  - Learns to combine frozen + new information
  
Consolidation:
  - Symbols 10-12: PROTECTED (never tested/removed)
  - Symbols 13-15: Can be tested and removed if weak
```

## Key Benefits

1. **Frozen Symbols Are Truly Frozen**
   - Never retrained in pretraining
   - Never tested in consolidation
   - Never removed
   - Not assigned to new puzzles (fresh puzzles get NEW symbols)

2. **Clean Phase Separation**
   - Phase 0 symbols remain stable
   - Phase 1 adds NEW symbols for NEW information
   - No interference between phases

3. **Correct Progressive Learning**
   - Position 0 uses known, stable symbols
   - Position 1 learns new symbols
   - System builds incrementally

4. **No Wasted Computation**
   - Pretraining skips frozen puzzles
   - Training focuses on new symbols
   - Consolidation only analyzes trainable symbols

## Symbol Assignment Flow

### Phase 0: Fresh Start (No Snapshot)
```
Symbols:     [10] [11] [12] [13] [14] ...
Puzzles:      0    1    2    -    -
Status:      NEW  NEW  NEW  AVL  AVL
Pretraining: ✓    ✓    ✓
Training:    ✓    ✓    ✓    (position 0)
Consolidate: ✓    ✓    ✓
```

### Phase 1: Resume from Snapshot
```
Symbols:     [10] [11] [12] [13] [14] [15] ...
Puzzles:      -    -    -    3    4    5
Status:      FRZ  FRZ  FRZ  NEW  NEW  NEW
Pretraining: ✗    ✗    ✗    ✓    ✓    ✓
Training:    ✓    ✓    ✓    ✓    ✓    ✓   (pos 0: 10-12, pos 1: 13-15)
Consolidate: ✗    ✗    ✗    ✓    ✓    ✓
```

Legend:
- **FRZ**: Frozen (from previous phase)
- **NEW**: Newly added in this phase
- **AVL**: Available for assignment
- **✓**: Can be trained/tested/removed
- **✗**: Protected (no training/testing/removal)

## Testing

To verify frozen symbols are protected:

1. **Check Pretraining Logs**
   ```
   Skipped N puzzles mapped to frozen symbols (symbols X-Y)
   ```

2. **Check Consolidation Logs**
   ```
   Protected frozen symbols: 10-12 (will not be removed)
   Symbol 10: FROZEN (skipping consolidation analysis)
   ```

3. **Check Gradients** (during training)
   - Frozen symbol embeddings should have zero gradients
   - Frozen position predictors should have no gradient updates

## Summary

**Frozen symbols from previous training phases are now completely protected throughout all subsequent training:**

- ✅ Embeddings frozen (gradient hook zeros gradients)
- ✅ Not assigned to new puzzles (fresh puzzles get NEW symbols)
- ✅ Not retrained in pretraining (new puzzles train NEW symbols)
- ✅ Not tested in consolidation
- ✅ Not removed in consolidation
- ✅ Only used during training (in frozen positions)

**Simplified Approach:**
- Each phase gets **fresh puzzles** assigned to **NEW symbols**
- Frozen symbols from previous phases are only used during training (in their frozen positions)
- No need to restore old puzzle mappings - they're not used

This ensures true progressive learning where each phase builds on stable foundations from previous phases! 🎯

