# Snapshot Selection Feature - Web UI

## Overview

Added a dropdown menu at the top of the configuration panel that allows you to select a saved snapshot to resume training from. When a snapshot is selected, training automatically continues with frozen positions and communication symbols from that snapshot.

## UI Location

The snapshot selector appears at the top of the Configuration section, above the tabs:

```
Configuration
┌─────────────────────────────────────────────────┐
│ 📦 Resume from Snapshot              [🔄 Refresh] │
│ [Dropdown: No snapshot (start fresh)        ▼] │
│ Snapshot Info:                                  │
│ 📅 Created: 20250101_120000                     │
│ 🔧 Architecture: 512d embed, 10 max seq        │
│ 📊 Training State: seq_len=1, comm_symbols=9   │
│ 🔒 Frozen positions: []                         │
│ 🔒 Frozen comm symbols: 0                       │
└─────────────────────────────────────────────────┘
  [Basic] [Advanced] [Architecture]
```

## Features

### 1. **Snapshot Dropdown**
- Lists all available snapshots from `outputs/snapshots/`
- Sorted with newest first
- Default option: "No snapshot (start fresh)"
- Auto-refreshes after saving a new snapshot

### 2. **Snapshot Info Display**
When you select a snapshot, it automatically displays:
- **Created**: Timestamp of snapshot creation
- **Name**: Custom label if provided
- **Architecture**: Embedding dimensions, max sequence length
- **Training State**: Current sequence length, number of communication symbols
- **Frozen Positions**: Which positions were frozen
- **Frozen Comm Symbols**: How many symbols were frozen

### 3. **Refresh Button** (🔄)
- Manually refresh the snapshot list
- Useful after saving snapshots outside the web UI

### 4. **Automatic Integration**
- Selected snapshot is sent to training script on start
- Script loads snapshot and applies freezing automatically
- No manual command-line arguments needed

## Workflow

### Phase 0: Train from Scratch

1. **Configuration Panel**
   - Resume from Snapshot: "No snapshot (start fresh)"
   - current_seq_length: 1
   - initial_comm_symbols: 4

2. **Train**
   - Click "Start Training"
   - System trains with 1-symbol messages
   - Learns ~9 communication symbols through addition/consolidation

3. **Save Snapshot**
   - Click "Save Snapshot" button
   - Enter optional name (e.g., "phase0_seq1")
   - Snapshot saved to `outputs/snapshots/`
   - Dropdown automatically refreshes after 2 seconds

### Phase 1: Resume from Snapshot

1. **Select Snapshot**
   - Open "Resume from Snapshot" dropdown
   - Select "phase0_seq1_20250101_120000.pt"
   - Info automatically displays:
     ```
     📅 Created: 20250101_120000
     🏷️ Name: phase0_seq1
     🔧 Architecture: 512d embed, 10 max seq
     📊 Training State: seq_len=1, comm_symbols=9
     🔒 Frozen positions: []
     🔒 Frozen comm symbols: 0
     ```

2. **Update Configuration**
   - current_seq_length: 2 (expand to 2-symbol messages)
   - Keep other settings as needed

3. **Start Training**
   - Click "Start Training"
   - System automatically:
     - Loads snapshot weights
     - Freezes position 0 predictor
     - Freezes communication symbols 10-18 (the 9 learned symbols)
     - Sets active sequence length to 2
   - Training uses:
     - Position 0: Frozen symbols (10-18)
     - Position 1: New trainable symbols (19+)

4. **Monitor Progress**
   - Console shows freezing details:
     ```
     ═══════════════════════════════════════
     Loading snapshot from: outputs/snapshots/phase0_seq1_20250101_120000.pt
     ═══════════════════════════════════════
     ✓ Loaded agent state dictionaries
     
     Snapshot Architecture:
       embedding_dim: 512
       hidden_dim: 1024
       num_symbols: 100
       puzzle_symbols: 10
       max_seq_length: 10
       similarity_metric: cosine
     
     ────────────────────────────────────────
     Applying Freezing from Snapshot:
     ────────────────────────────────────────
       • No frozen positions (first phase)
       • Frozen communication symbols: 9
         (indices 10 to 18)
     ────────────────────────────────────────
     
     ✓ Updating sequence length: 1 → 2
     ✓ Auto-freezing positions for previous sequence length: [0]
     [sender] Froze position 0 predictor
     [receiver] Froze position 0 predictor
     
     ✓ Successfully resumed from snapshot
     ═══════════════════════════════════════
     ```

5. **Save Phase 1 Snapshot**
   - After training completes, save another snapshot
   - This becomes the starting point for Phase 2 (seq_len=3)

## Technical Implementation

### Frontend (script.js)

```javascript
// Snapshot state tracking
let currentSnapshots = [];
let selectedSnapshot = null;

// Load snapshots from server
async function loadSnapshots() {
    const response = await fetch('/api/snapshot/list');
    const data = await response.json();
    currentSnapshots = data.snapshots;
    populateSnapshotDropdown();
}

// Handle snapshot selection
async function onSnapshotSelected() {
    const select = document.getElementById('resume_snapshot');
    selectedSnapshot = select.value || null;
    
    if (selectedSnapshot) {
        // Fetch and display snapshot metadata
        const response = await fetch('/api/snapshot/inspect', {
            method: 'POST',
            body: JSON.stringify({ filename: selectedSnapshot })
        });
        // Display architecture and training state
    }
}

// Include snapshot when starting training
async function startTraining() {
    const requestBody = {
        resume_from: selectedSnapshot || null
    };
    
    await fetch('/api/start', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });
}
```

### Backend (web_app.py)

```python
@app.route('/api/start', methods=['POST'])
def start_training():
    # Get selected snapshot from request
    request_data = request.get_json(force=True) or {}
    resume_from = request_data.get('resume_from')
    
    if resume_from:
        # Validate snapshot exists
        snap_dir = os.path.join(output_dir, 'snapshots')
        snapshot_path = os.path.join(snap_dir, resume_from)
        
        if not os.path.exists(snapshot_path):
            return jsonify({'error': 'Snapshot not found'})
    
    # Build command
    cmd = [
        sys.executable, 'src/train_selection.py',
        '--config', 'training_config.json',
        '--web-mode'
    ]
    
    # Add --resume-from if snapshot selected
    if resume_from:
        cmd.extend(['--resume-from', snapshot_path])
    
    # Start process
    subprocess.Popen(cmd, ...)
```

### Training Script (train_selection.py)

The script already handles the `--resume-from` argument:
1. Loads agent state dictionaries
2. Applies position freezing
3. Applies communication symbol freezing
4. Updates sequence length if needed
5. Auto-freezes previous positions when expanding

## Benefits

1. **No Command-Line Required**: Everything through the web UI
2. **Visual Feedback**: See snapshot metadata before starting
3. **Error Prevention**: Validates snapshot exists before starting
4. **Automatic Updates**: Snapshot list refreshes after saving
5. **Seamless Progressive Training**: Just select, configure, and start

## Notes

- Snapshots are stored in `outputs/snapshots/` by default
- Snapshot files are named: `{label}_{timestamp}.pt`
- If no snapshot selected, training starts fresh (same as before)
- Selected snapshot persists in UI during page session
- Refresh button useful if snapshots added externally

## Example Progressive Training Sequence

```
Phase 0 (seq=1):
  └─ Select: "No snapshot"
  └─ Train → Save → "phase0.pt"

Phase 1 (seq=2):
  └─ Select: "phase0.pt"
  └─ Update: current_seq_length=2
  └─ Train → Save → "phase1.pt"
  
Phase 2 (seq=3):
  └─ Select: "phase1.pt"
  └─ Update: current_seq_length=3
  └─ Train → Save → "phase2.pt"

Phase 3 (seq=4):
  └─ Select: "phase2.pt"
  └─ Update: current_seq_length=4
  └─ Train...
```

Each phase builds on the previous, with proper freezing applied automatically!

