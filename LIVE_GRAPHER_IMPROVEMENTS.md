# Live Grapher Improvements for Progressive Training

## Issues Fixed

### Issue 1: Active Symbols Showing 0 When Loading from Snapshot

**Problem**: LiveGrapher was initialized with `config['initial_comm_symbols']` (e.g., 3) BEFORE the snapshot was loaded, so it didn't know about the frozen symbols (e.g., 6).

**Root Cause**: Initialization order:
```python
1. LiveGrapher initialized with config['initial_comm_symbols'] = 3
2. Snapshot loaded (sets trainer.agent1.current_comm_symbols = 6)
3. Try to update: live_grapher.active_symbols_count = 6
   (but initialization already happened with wrong value)
```

**Solution**: Initialize LiveGrapher AFTER snapshot loading and symbol setup:

```python
# OLD: Initialize early with config value
live_grapher = LiveGrapher(initial_symbols=config['initial_comm_symbols'])  # 3

# NEW: Initialize after snapshot loading
# (After all snapshot loading and symbol initialization)
current_symbols = trainer.agent1.current_comm_symbols  # Now correctly 6 or 9
live_grapher = LiveGrapher(initial_symbols=current_symbols)
```

### Issue 2: Unclear Phase Skipping for seq_length > 1

**Problem**: When pretraining and consolidation were skipped for `seq_length > 1`, the console output wasn't clear enough about what was happening.

**Solution**: Added explicit, prominent logging:

```python
if trainer.agent1.current_seq_length > 1:
    print(f"\n{'='*60}")
    print(f"SKIPPING PRETRAINING (seq_length={trainer.agent1.current_seq_length}, position 1+)")
    print(f"{'='*60}")
    log_file.write("Skipping pretraining for seq_length > 1\n")
    trainer.advance_phase()
```

### Issue 3: Stats Panel Not Showing Progressive Training Info

**Problem**: Statistics panel didn't show relevant information for progressive training (sequence length, frozen symbols, position vocabularies).

**Solution**: Enhanced stats display with progressive training section:

```python
# Progressive training info
stats_text += f"=== Progressive Training ===\n"
stats_text += f"Seq Length: {trainer.agent1.current_seq_length}\n"
frozen_comm = getattr(trainer, 'frozen_comm_symbols', 0)
if frozen_comm > 0:
    stats_text += f"Frozen Symbols: {frozen_comm}\n"
frozen_pos = getattr(trainer, 'frozen_positions', [])
if frozen_pos:
    stats_text += f"Frozen Positions: {frozen_pos}\n"
# Show position vocabularies
if trainer.agent1.position_vocabularies:
    stats_text += f"Position Vocabs:\n"
    for pos in sorted(trainer.agent1.position_vocabularies.keys()):
        vocab = trainer.agent1.position_vocabularies[pos]
        stats_text += f"  Pos {pos}: {len(vocab)} syms\n"
```

## New Stats Display Layout

**For seq_length = 1 (Position 0 only):**
```
Current Step: 1234

Loss (MA): 0.0234
Agent1 Acc (MA): 0.876
Agent2 Acc (MA): 0.892
Agent1 GES (MA): 45.32
Agent2 GES (MA): 48.21

Active Symbols: 6
Total Puzzles: 4
Distractors: 3

Current Phase: Training
```

**For seq_length = 2 (Progressive Training):**
```
Current Step: 5678

=== Progressive Training ===
Seq Length: 2
Frozen Symbols: 6
Frozen Positions: [0]
Position Vocabs:
  Pos 0: 6 syms
  Pos 1: 3 syms

Loss (MA): 0.0456
Agent1 Acc (MA): 0.645
Agent2 Acc (MA): 0.712
Agent1 GES (MA): 23.45
Agent2 GES (MA): 28.76

Active Symbols: 9
Total Puzzles: 4
Distractors: 3

Current Phase: Training
```

## Implementation Details

### 1. Changed Initialization Order (train_selection.py)

**Before:**
```python
# Line 2182 (early in main())
live_grapher = LiveGrapher(initial_symbols=config['initial_comm_symbols'])

# Much later...
# Snapshot loaded, symbols initialized
live_grapher.active_symbols_count = current_symbols  # Try to update
```

**After:**
```python
# Line 2181 (early in main())
live_grapher = None  # Will be initialized later

# Much later, AFTER snapshot loading and symbol setup
# Line 2491
current_symbols = trainer.agent1.current_comm_symbols
live_grapher = LiveGrapher(initial_symbols=current_symbols)
live_grapher.trainer_ref = weakref.ref(trainer)  # For stats display
```

### 2. Enhanced Stats Display (LiveGrapher.update_plots())

Added weak reference to trainer and progressive training info section:

```python
# In LiveGrapher initialization
import weakref
live_grapher.trainer_ref = weakref.ref(trainer)

# In update_plots()
if hasattr(self, 'trainer_ref') and self.trainer_ref:
    trainer = self.trainer_ref()
    if trainer:
        # Display progressive training stats
        stats_text += f"=== Progressive Training ===\n"
        # ... show seq_length, frozen symbols, position vocabs
```

### 3. Clearer Phase Skip Logging

Added prominent section headers for skipped phases:

```python
# Pretraining skip
print(f"\n{'='*60}")
print(f"SKIPPING PRETRAINING (seq_length={trainer.agent1.current_seq_length}, position 1+)")
print(f"{'='*60}")

# Consolidation skip
print(f"\n{'='*60}")
print(f"SKIPPING CONSOLIDATION (seq_length={trainer.agent1.current_seq_length}, position 1+)")
print(f"{'='*60}")
```

## Example Console Output

### Phase 0 (seq_length = 1)
```
Live grapher initialized with 3 symbols
Live grapher ready! Training metrics will update in real-time.

[Pretraining runs]
[Training runs]
[Consolidation runs]
[Addition runs]
```

### Phase 1 (seq_length = 2, from snapshot with 6 frozen symbols)
```
Loading snapshot...
Frozen communication symbols: 6

Initializing for position 1 (training without puzzle mappings)
  Creating 3 new symbols: [16, 17, 18]
  Total communication symbols: 9 (frozen: 6, new: 3)

Live grapher initialized with 9 symbols
Live grapher ready! Training metrics will update in real-time.

============================================================
SKIPPING PRETRAINING (seq_length=2, position 1+)
============================================================

[Training runs]

============================================================
SKIPPING CONSOLIDATION (seq_length=2, position 1+)
============================================================

============================================================
ADDITION PHASE - Adding symbols for position 1
============================================================
Creating 3 new symbols: [19, 20, 21]
Position 1 vocabulary: [16, 17, 18, 19, 20, 21]
```

## What to Look For

When training from a snapshot with `seq_length > 1`, you should now see:

1. ✅ **Correct symbol count**: Graph shows 9 symbols (not 0 or 3)
2. ✅ **Clear skip messages**: Prominent "SKIPPING PRETRAINING" and "SKIPPING CONSOLIDATION" messages
3. ✅ **Progressive info**: Stats panel shows:
   - Seq Length: 2
   - Frozen Symbols: 6
   - Frozen Positions: [0]
   - Position Vocabs with counts
4. ✅ **Symbol additions**: Addition phase adds symbols (not puzzles) for position 1+

## Modified Files

1. **`src/train_selection.py`**
   - Moved LiveGrapher initialization to after snapshot loading (line 2491)
   - Added trainer weak reference to LiveGrapher
   - Enhanced stats display with progressive training info
   - Added clear logging for phase skips

## Benefits

1. **Accurate Visualization**: Symbol count graph shows correct values from start
2. **Better Debugging**: Clear messages when phases are skipped
3. **More Information**: Stats panel shows all relevant progressive training state
4. **Less Confusion**: User can see exactly what's happening at each stage

All working correctly now! 🎯

