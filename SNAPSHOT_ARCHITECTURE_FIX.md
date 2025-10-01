# Snapshot Architecture Compatibility Fix

## Problem

When loading a snapshot, the model was being created with architecture parameters from the **config file** first, then attempting to load weights from the snapshot. This caused errors when the snapshot had different architecture parameters:

```
Error(s) in loading state_dict for ProgressiveSelectionAgent:
Unexpected key(s) in state_dict: "encoder.position_predictors.5.0.weight", ...
size mismatch for encoder.position_temperatures: copying a param with shape 
torch.Size([20]) from checkpoint, the shape in current model is torch.Size([5]).
```

**Root Cause**: Snapshot was saved with `max_seq_length=20`, but config file had `max_seq_length=5`.

## Solution

Modified the snapshot loading sequence to:
1. **Load snapshot FIRST** (before creating agents)
2. **Extract architecture parameters** from snapshot
3. **Override config** with snapshot architecture
4. **Create agents** with snapshot architecture
5. **Load state dicts** (now compatible!)
6. **Apply freezing**

## Implementation

### New Loading Flow in `train_selection.py`

```python
# 1. Load snapshot and extract architecture BEFORE creating agents
snapshot_data = None
if args.resume_from:
    snapshot_data = torch.load(args.resume_from, map_location=device)
    
    if 'architecture' in snapshot_data:
        arch = snapshot_data['architecture']
        
        # Override config with snapshot architecture
        arch_keys = ['embedding_dim', 'hidden_dim', 'num_symbols', 
                     'puzzle_symbols', 'max_seq_length']
        for key in arch_keys:
            if key in arch:
                old_val = config.get(key)
                new_val = arch[key]
                if old_val != new_val:
                    print(f"  {key}: {old_val} → {new_val}")
                    config[key] = new_val

# 2. Create agents with (potentially overridden) config
sender = Agent(
    agent_id="sender",
    embedding_dim=config['embedding_dim'],    # From snapshot!
    hidden_dim=config['hidden_dim'],          # From snapshot!
    num_symbols=config['num_symbols'],        # From snapshot!
    puzzle_symbols=config['puzzle_symbols'],  # From snapshot!
    max_seq_length=config['max_seq_length'],  # From snapshot!
    ...
).to(device)

# 3. Now load state dict (architecture matches!)
if snapshot_data is not None:
    trainer.agent1.load_state_dict(snapshot_data['agent1_state_dict'])
    trainer.agent2.load_state_dict(snapshot_data['agent2_state_dict'])
    # Apply freezing...
```

## Architecture Parameters Locked from Snapshot

When resuming from snapshot, these parameters are **automatically locked** to match the snapshot:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `embedding_dim` | Embedding dimension | 512 |
| `hidden_dim` | Hidden layer dimension | 1024 |
| `num_symbols` | Maximum total symbols | 100 |
| `puzzle_symbols` | Number of puzzle symbols | 10 |
| `max_seq_length` | Maximum sequence length (architecture capacity) | 20 |

## Console Output

When loading a snapshot, you'll now see:

```
============================================================
Loading snapshot state from: ./outputs/snapshots/yes_20250930_180939.pt
============================================================

Snapshot Architecture:
  embedding_dim: 512
  hidden_dim: 1024
  num_symbols: 100
  puzzle_symbols: 10
  max_seq_length: 20
  similarity_metric: cosine

⚠️  Overriding config with snapshot architecture parameters:
  embedding_dim: 512 (unchanged)
  hidden_dim: 1024 (unchanged)
  num_symbols: 100 (unchanged)
  puzzle_symbols: 10 (unchanged)
  max_seq_length: 5 → 20          ← OVERRIDDEN!

Creating agents with:
  embedding_dim: 512
  hidden_dim: 1024
  num_symbols: 100
  puzzle_symbols: 10
  max_seq_length: 20               ← Uses snapshot value

============================================================
Loading Weights and Applying Freezing
============================================================

✓ Loaded agent state dictionaries

────────────────────────────────────────
Applying Freezing from Snapshot:
────────────────────────────────────────
  • No frozen positions (first phase)
  • Frozen communication symbols: 9
    (indices 10 to 18)
────────────────────────────────────────

✓ Successfully loaded snapshot and applied freezing
============================================================
```

## Trainable Parameters

After loading, you can **still modify** non-architecture parameters:

| Parameter | Can Override? | Notes |
|-----------|---------------|-------|
| `current_seq_length` | ✅ Yes | Expand sequence length for next phase |
| `learning_rate` | ✅ Yes | Adjust learning rate |
| `num_distractors` | ✅ Yes | Change distractor count |
| `training_cycles` | ✅ Yes | Adjust training length |
| `max_seq_length` | ❌ No | Locked by snapshot architecture |
| `embedding_dim` | ❌ No | Locked by snapshot architecture |
| `hidden_dim` | ❌ No | Locked by snapshot architecture |

## Example Usage

### Phase 0: Initial Training
```json
// training_config.json
{
  "max_seq_length": 20,        // Architecture capacity
  "current_seq_length": 1,     // Start with 1 position
  "embedding_dim": 512,
  "hidden_dim": 1024,
  ...
}
```

Train and save snapshot → `phase0.pt`

### Phase 1: Resume with Wrong Config
```json
// training_config.json (accidentally changed)
{
  "max_seq_length": 5,         // ❌ WRONG! But will be auto-fixed
  "current_seq_length": 2,     // ✅ OK to change
  "embedding_dim": 512,
  "hidden_dim": 1024,
  ...
}
```

**Before Fix**: ❌ Would crash with size mismatch errors

**After Fix**: ✅ Automatically detects and fixes:
```
⚠️  Overriding config with snapshot architecture parameters:
  max_seq_length: 5 → 20
```

System continues with correct architecture!

## Benefits

1. **Fool-proof**: Can't accidentally use wrong architecture
2. **Clear Feedback**: Console shows all parameter overrides
3. **Backward Compatible**: Works with old snapshots (uses config if no architecture metadata)
4. **Flexible**: Can still modify training parameters (learning rate, cycles, etc.)
5. **Safe**: Architecture parameters locked to prevent model corruption

## Error Handling

### If Snapshot Missing Architecture Metadata
```
⚠️  Warning: Snapshot does not contain architecture metadata
Using config file architecture (may cause loading errors)
```

### If Architecture Loading Fails
```
✗ Error loading snapshot architecture: <error details>
Continuing with config file architecture...
```

### If Weight Loading Fails
```
✗ Error loading snapshot weights: <error details>
Continuing with fresh initialization...
```

In all cases, training continues (either with snapshot or fresh start).

## Summary

**The fix ensures that when you resume from a snapshot, the model architecture ALWAYS matches the snapshot, preventing any shape mismatch errors.** Your config file's architecture parameters are automatically overridden to match the snapshot, while training parameters (like `current_seq_length`, learning rate, etc.) remain configurable.

This makes progressive training much more robust and user-friendly! 🎯

