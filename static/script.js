// Global variables
let currentConfig = {};
let autoScroll = true;
let statusInterval;
let logsInterval;

let showScoreHistory = false;

// Live plot auto-refresh
let livePlotInterval;
let scoresInterval;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadConfig();
    loadSnapshots();  // Load available snapshots
    startStatusUpdates();
    startLogUpdates();
    // Keep manual refresh around but start live updates automatically
    startLivePlotUpdates();
    startScoresUpdates();
    initTrainingModeControls();
    initReconControls();
    startReconUpdates();
    startSelectionUpdates();
});

// Configuration Management
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            currentConfig = await response.json();
            populateConfigForm();
        }
    } catch (error) {
        console.error('Error loading config:', error);
        showNotification('Error loading configuration', 'error');
    }
}

function populateConfigForm() {
    for (const [key, value] of Object.entries(currentConfig)) {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    }
    // Also update run title display if present
    const display = document.getElementById('run-title-display');
    if (display && currentConfig.run_title) {
        display.textContent = currentConfig.run_title;
    }
}

async function saveConfig() {
    // Collect form data
    const config = {};
    const formElements = [
        'max_global_phases', 'initial_puzzle_count', 'training_cycles', 
        'first_training_cycles', 'first_pretrain_epochs', 'pretrain_epochs',
        'puzzles_per_addition', 'learning_rate', 'num_distractors',
        'distractor_strategy', 'phase_change_indicator', 'early_stop_min_cycles', 
        'consolidation_threshold', 'embedding_dim', 'hidden_dim', 'num_symbols', 'puzzle_symbols',
        'max_seq_length', 'current_seq_length',
        // NEW: run title
        'run_title'
    ];
    
    formElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            let value = element.value;
            // Convert numbers
            if (element.type === 'number') {
                value = parseFloat(value);
            }
            config[id] = value;
        }
    });
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            currentConfig = await response.json();
            showNotification('Configuration saved successfully', 'success');
        } else {
            showNotification('Error saving configuration', 'error');
        }
    } catch (error) {
        console.error('Error saving config:', error);
        showNotification('Error saving configuration', 'error');
    }
}

// Snapshot Management
let currentSnapshots = [];
let selectedSnapshot = null;

async function loadSnapshots() {
    try {
        const response = await fetch('/api/snapshot/list');
        if (response.ok) {
            const data = await response.json();
            currentSnapshots = data.snapshots || [];
            populateSnapshotDropdown();
        }
    } catch (error) {
        console.error('Error loading snapshots:', error);
    }
}

function populateSnapshotDropdown() {
    const select = document.getElementById('resume_snapshot');
    
    // Clear existing options except the first (default)
    while (select.options.length > 1) {
        select.remove(1);
    }
    
    // Add snapshot options (newest first)
    currentSnapshots.forEach(snapshot => {
        const option = document.createElement('option');
        option.value = snapshot;
        option.textContent = snapshot;
        select.appendChild(option);
    });
    
    // Restore previously selected snapshot if it still exists
    if (selectedSnapshot && currentSnapshots.includes(selectedSnapshot)) {
        select.value = selectedSnapshot;
        onSnapshotSelected();
    }
}

async function refreshSnapshots() {
    await loadSnapshots();
    showNotification('Snapshot list refreshed', 'success');
}

async function onSnapshotSelected() {
    const select = document.getElementById('resume_snapshot');
    selectedSnapshot = select.value || null;
    const infoDiv = document.getElementById('snapshot-info');
    
    if (!selectedSnapshot) {
        infoDiv.innerHTML = '';
        return;
    }
    
    // Show loading state
    infoDiv.innerHTML = '<em>Loading snapshot info...</em>';
    
    try {
        // Inspect the snapshot to show metadata
        const response = await fetch('/api/snapshot/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: selectedSnapshot })
        });
        
        if (response.ok) {
            const data = await response.json();
            const arch = data.architecture || {};
            const state = data.trainer_state || {};
            const meta = data.meta || {};
            
            let info = `<strong>Snapshot Info:</strong><br>`;
            if (meta.created_at) {
                info += `📅 Created: ${meta.created_at}<br>`;
            }
            if (meta.name) {
                info += `🏷️ Name: ${meta.name}<br>`;
            }
            info += `🔧 Architecture: ${arch.embedding_dim || '?'}d embed, ${arch.max_seq_length || '?'} max seq<br>`;
            info += `📊 Training State: seq_len=${state.current_seq_length || '?'}, `;
            info += `comm_symbols=${state.current_comm_symbols_a1 || '?'}<br>`;
            
            if (state.frozen_positions && state.frozen_positions.length > 0) {
                info += `🔒 Frozen positions: ${state.frozen_positions.join(', ')}<br>`;
            }
            if (state.frozen_comm_symbols > 0) {
                info += `🔒 Frozen comm symbols: ${state.frozen_comm_symbols}<br>`;
            }
            
            infoDiv.innerHTML = info;
        } else {
            infoDiv.innerHTML = '<em style="color: #c33;">Error loading snapshot info</em>';
        }
    } catch (error) {
        console.error('Error inspecting snapshot:', error);
        infoDiv.innerHTML = '<em style="color: #c33;">Error: ' + error.message + '</em>';
    }
}

function resetConfig() {
    if (confirm('Reset all configuration to defaults?')) {
        // Reset to default values
        const defaults = {
            max_global_phases: 100,
            initial_puzzle_count: 4,
            training_cycles: 25,
            first_training_cycles: 50,
            puzzles_per_addition: 3,
            learning_rate: 0.0000007,
            num_distractors: 3,
            distractor_strategy: 'random',
            phase_change_indicator: 'ges',
            early_stop_min_cycles: 5,
            consolidation_threshold: 0.3,
            embedding_dim: 512,
            hidden_dim: 1024,
            num_symbols: 100,
            puzzle_symbols: 10
        };
        
        for (const [key, value] of Object.entries(defaults)) {
            const element = document.getElementById(key);
            if (element) {
                element.value = value;
            }
        }
        
        showNotification('Configuration reset to defaults', 'info');
    }
}

// Training Control
async function startTraining() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    // Save current config before starting
    await saveConfig();
    
    try {
        // Include selected snapshot if any
        const requestBody = {
            resume_from: selectedSnapshot || null
        };
        
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            showNotification('Training started successfully', 'success');
            
            // Update UI to show running state
            document.getElementById('training-status').textContent = 'Running';
            document.getElementById('training-status').classList.add('running');
            const display = document.getElementById('run-title-display');
            if (display) {
                const titleInput = document.getElementById('run_title');
                display.textContent = titleInput && titleInput.value ? titleInput.value : (currentConfig.run_title || '');
            }
            
            // Clear score UI for new run
            clearScoreUI();
        } else {
            showNotification(result.message || 'Error starting training', 'error');
        }
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Error starting training', 'error');
    }
}

function clearScoreUI() {
    document.getElementById('unseen-score').textContent = 'No data yet';
    document.getElementById('novel-score').textContent = 'No data yet';
    const uh = document.getElementById('unseen-history');
    const nh = document.getElementById('novel-history');
    if (uh) uh.innerHTML = '';
    if (nh) nh.innerHTML = '';
}

async function stopTraining() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    if (confirm('Are you sure you want to stop training?')) {
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                showNotification('Training stopped', 'info');
                
                // Update UI to show stopped state
                document.getElementById('training-status').textContent = 'Stopped';
                document.getElementById('training-status').classList.remove('running');
            } else {
                showNotification(result.message || 'Error stopping training', 'error');
            }
        } catch (error) {
            console.error('Error stopping training:', error);
            showNotification('Error stopping training', 'error');
        }
    }
}

// Status Updates
function startStatusUpdates() {
    statusInterval = setInterval(updateStatus, 2000); // Update every 2 seconds
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        // Update buttons
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        startBtn.disabled = status.running;
        stopBtn.disabled = !status.running;
        
        // Update status display
        const trainingStatus = document.getElementById('training-status');
        if (status.running) {
            trainingStatus.textContent = 'Running';
            trainingStatus.classList.add('running');
        } else {
            trainingStatus.textContent = 'Idle';
            trainingStatus.classList.remove('running');
        }
        
        // Update detailed status
        if (status.status_info) {
            const info = status.status_info;
            document.getElementById('current-phase').textContent = info.current_phase || 'Unknown';
            document.getElementById('progress-detail').textContent = `${info.progress || 0}%`;
            document.getElementById('status-message').textContent = info.message || 'No message';
            const titleDisplay = document.getElementById('run-title-display');
            if (titleDisplay) {
                titleDisplay.textContent = (info.run_title || currentConfig.run_title || '').trim();
            }
            
            // Update progress bar
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            const progress = info.progress || 0;
            
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
        }
        
        document.getElementById('process-pid').textContent = status.pid || 'None';
        // Update training mode display
        refreshTrainingModeUI();
        
    } catch (error) {
        console.error('Error updating status:', error);
        document.getElementById('connection-status').textContent = 'Connection Error';
    }
}

// --- Runtime Training Mode Toggle ---
async function initTrainingModeControls() {
    try {
        await refreshTrainingModeUI();
        const btn = document.getElementById('toggle-mode-btn');
        if (btn) {
            btn.addEventListener('click', toggleTrainingMode);
        }
    } catch (e) {
        console.error('initTrainingModeControls error', e);
    }
}

async function refreshTrainingModeUI() {
    try {
        const resp = await fetch('/api/training-mode');
        if (!resp.ok) return;
        const data = await resp.json();
        const mode = (data.mode || 'selection');
        const modeLabel = document.getElementById('training-mode-label');
        const btn = document.getElementById('toggle-mode-btn');
        if (modeLabel) modeLabel.textContent = mode === 'reconstruction' ? 'Reconstruction' : 'Selection';
        if (btn) btn.textContent = mode === 'reconstruction' ? 'Switch to Selection' : 'Switch to Reconstruction';
    } catch (e) {
        // silent
    }
}

async function toggleTrainingMode() {
    try {
        const current = await (await fetch('/api/training-mode')).json();
        const mode = current.mode === 'reconstruction' ? 'selection' : 'reconstruction';
        const resp = await fetch('/api/training-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showNotification(err.error || 'Failed to change mode', 'error');
            return;
        }
        await refreshTrainingModeUI();
        showNotification(`Training mode set to ${mode}`, 'success');
    } catch (e) {
        console.error('toggleTrainingMode error', e);
        showNotification('Error toggling mode', 'error');
    }
}

// Log Updates
function startLogUpdates() {
    logsInterval = setInterval(updateLogs, 3000); // Update every 3 seconds
}

async function updateLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        
        const logOutput = document.getElementById('log-output');
        const logContainer = document.getElementById('log-container');
        
        // Clear and populate logs
        logOutput.innerHTML = '';
        
        data.logs.forEach(log => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${log.type}`;
            
            const timestamp = new Date(log.timestamp * 1000).toLocaleTimeString();
            logEntry.textContent = `[${timestamp}] ${log.message}`;
            
            logOutput.appendChild(logEntry);
        });
        
        // Auto-scroll to bottom if enabled
        if (autoScroll) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
    } catch (error) {
        console.error('Error updating logs:', error);
    }
}

function clearLogs() {
    document.getElementById('log-output').innerHTML = '';
    showNotification('Logs cleared', 'info');
}

// --- Snapshot save ---
async function saveSnapshot() {
    try {
        const defaultName = new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
        const name = prompt('Optional label for snapshot (leave blank for timestamp):', defaultName) || '';
        const resp = await fetch('/api/snapshot/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showNotification(err.error || 'Failed to request snapshot', 'error');
            return;
        }
        showNotification('Snapshot requested. It will be saved shortly.', 'success');
        
        // Refresh snapshot list after a short delay (to allow snapshot to be written)
        setTimeout(async () => {
            await loadSnapshots();
        }, 2000);
    } catch (e) {
        console.error('saveSnapshot error', e);
        showNotification('Error requesting snapshot', 'error');
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    document.getElementById('autoscroll-status').textContent = autoScroll ? 'ON' : 'OFF';
    showNotification(`Auto-scroll ${autoScroll ? 'enabled' : 'disabled'}`, 'info');
}

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// Plot Management
async function refreshPlots() {
    try {
        const response = await fetch('/api/plots');
        const data = await response.json();
        
        const selector = document.getElementById('plot-selector');
        selector.innerHTML = '<option value="">Select a plot...</option>';
        
        // Sort plots to show newest first (by name if timestamped)
        const plots = (data.plots || []).slice().sort().reverse();
        
        plots.forEach(plot => {
            const option = document.createElement('option');
            option.value = plot;
            option.textContent = plot;
            selector.appendChild(option);
        });
        
        // Auto-select stable phase plot if present, else most recent
        const preferred = plots.find(p => p.endsWith('phase_training_metrics.png')) || plots[0];
        if (preferred) {
            selector.value = preferred;
            showPlot();
        }
        
    } catch (error) {
        console.error('Error refreshing plots:', error);
        showNotification('Error loading plots', 'error');
    }
}

function showPlot() {
    const selector = document.getElementById('plot-selector');
    const plotDisplay = document.getElementById('plot-display');
    
    if (selector.value) {
        const img = document.createElement('img');
        // value might include a directory; server expects path but enforces basename.
        img.src = `/api/plot/${encodeURIComponent(selector.value)}`;
        img.alt = selector.value;
        img.onerror = function() {
            plotDisplay.innerHTML = '<p>Error loading plot</p>';
        };
        
        plotDisplay.innerHTML = '';
        plotDisplay.appendChild(img);
    } else {
        plotDisplay.innerHTML = '<p>No plot selected</p>';
    }
}

// Live plot auto-refresh
function startLivePlotUpdates() {
    // Refresh every 5 seconds
    if (livePlotInterval) clearInterval(livePlotInterval);
    updateLivePlot();
    livePlotInterval = setInterval(updateLivePlot, 5000);
}

async function updateLivePlot() {
    try {
        const resp = await fetch('/api/plots/latest');
        const data = await resp.json();
        const plotDisplay = document.getElementById('plot-display');
        if (!data.latest) {
            plotDisplay.innerHTML = '<p>No plots available yet</p>';
            return;
        }
        // Ensure an img element exists
        let img = plotDisplay.querySelector('img');
        if (!img) {
            img = document.createElement('img');
            plotDisplay.innerHTML = '';
            plotDisplay.appendChild(img);
        }
        // Cache-bust using timestamp
        const src = `/api/plot/${encodeURIComponent(data.latest)}?t=${Date.now()}`;
        if (img.src !== src) {
            img.src = src;
            img.alt = data.latest;
        } else {
            // Force refresh by resetting src to new cache-busting query
            img.src = src;
        }
    } catch (e) {
        console.error('Error updating live plot:', e);
    }
}

// Scores auto-refresh
function startScoresUpdates() {
    if (scoresInterval) clearInterval(scoresInterval);
    updateScores();
    scoresInterval = setInterval(updateScores, 5000);
}

async function updateScores() {
    try {
        const resp = await fetch('/api/scores');
        const data = await resp.json();
        renderScore('unseen-score', data.unseen, 'Unseen');
        renderScore('novel-score', data.novel, 'Novel');
        renderScoreHistory('unseen-history', data.unseen_history, 'Unseen');
        renderScoreHistory('novel-history', data.novel_history, 'Novel');
        // Toggle visibility based on checkbox
        const display = showScoreHistory ? 'block' : 'none';
        const uh = document.getElementById('unseen-history');
        const nh = document.getElementById('novel-history');
        if (uh) uh.style.display = display;
        if (nh) nh.style.display = display;
    } catch (e) {
        console.error('Error updating scores:', e);
    }
}

function renderScore(elementId, score, label) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!score) {
        el.textContent = 'No data yet';
        return;
    }
    el.innerHTML = `
        <div><strong>Overall</strong>: ${score.correct}/${score.num_tests} (acc=${(score.accuracy*100).toFixed(1)}%)</div>
        <div><strong>A1→A2</strong>: ${score.a1_to_a2_correct ?? '-'} / ${score.a1_to_a2_total ?? '-'} (acc=${score.a1_to_a2_accuracy != null ? (score.a1_to_a2_accuracy*100).toFixed(1)+'%' : '-'})</div>
        <div><strong>A2→A1</strong>: ${score.a2_to_a1_correct ?? '-'} / ${score.a2_to_a1_total ?? '-'} (acc=${score.a2_to_a1_accuracy != null ? (score.a2_to_a1_accuracy*100).toFixed(1)+'%' : '-'})</div>
        <div><strong>GES (MA)</strong>: A1=${score.ges1_ma != null ? score.ges1_ma.toFixed(2) : '-'}, A2=${score.ges2_ma != null ? score.ges2_ma.toFixed(2) : '-'}</div>
    `;
}

function renderScoreHistory(elementId, history, label) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!history || history.length === 0) {
        el.innerHTML = '<div class="score-history-empty">No history</div>';
        return;
    }
    // Show most recent first
    const items = history.slice().reverse();
    const rows = items.map((s, idx) => {
        const overall = `${s.correct}/${s.num_tests} (${(s.accuracy*100).toFixed(1)}%)`;
        const a12 = (s.a1_to_a2_accuracy != null) ? `${(s.a1_to_a2_accuracy*100).toFixed(1)}%` : '-';
        const a21 = (s.a2_to_a1_accuracy != null) ? `${(s.a2_to_a1_accuracy*100).toFixed(1)}%` : '-';
        const ges1 = (s.ges1_ma != null) ? s.ges1_ma.toFixed(2) : '-';
        const ges2 = (s.ges2_ma != null) ? s.ges2_ma.toFixed(2) : '-';
        return `<div class="score-history-item">
            <span class="score-history-index">#${items.length - idx}</span>
            <span class="score-history-overall"><strong>Overall</strong>: ${overall}</span>
            <span class="score-history-a12"><strong>A1→A2</strong>: ${a12}</span>
            <span class="score-history-a21"><strong>A2→A1</strong>: ${a21}</span>
            <span class="score-history-ges"><strong>GES</strong>: A1=${ges1}, A2=${ges2}</span>
        </div>`;
    });
    el.innerHTML = rows.join('');
}

async function saveReport() {
    try {
        const resp = await fetch('/api/report');
        if (!resp.ok) {
            // Try to parse error JSON
            let msg = 'Failed to generate report';
            try {
                const err = await resp.json();
                if (err && err.error) msg = err.error;
            } catch (_) {}
            showNotification(msg, 'error');
            return;
        }
        const blob = await resp.blob();
        // Prefer server-provided filename
        const headerName = resp.headers.get('X-Report-Filename');
        const filename = headerName || `run_report_${new Date().toISOString().replace(/[:.]/g, '-')}.pdf`;
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showNotification('Report downloaded', 'success');
    } catch (e) {
        console.error('Report download error:', e);
        showNotification('Error generating report', 'error');
    }
}

// Notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.borderRadius = '5px';
    notification.style.zIndex = '1000';
    notification.style.maxWidth = '300px';
    notification.style.opacity = '0';
    notification.style.transform = 'translateX(100%)';
    notification.style.transition = 'all 0.3s ease';
    
    // Set colors based on type
    switch (type) {
        case 'success':
            notification.style.background = '#4CAF50';
            notification.style.color = 'white';
            break;
        case 'error':
            notification.style.background = '#f44336';
            notification.style.color = 'white';
            break;
        case 'warning':
            notification.style.background = '#ff9800';
            notification.style.color = 'white';
            break;
        default:
            notification.style.background = '#2196F3';
            notification.style.color = 'white';
    }
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+S to save config
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveConfig();
    }
    
    // Ctrl+Enter to start training
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const startBtn = document.getElementById('start-btn');
        if (!startBtn.disabled) {
            startTraining();
        }
    }
    
    // Escape to stop training
    if (e.key === 'Escape') {
        const stopBtn = document.getElementById('stop-btn');
        if (!stopBtn.disabled) {
            stopTraining();
        }
    }
});

function toggleScoreHistory() {
    showScoreHistory = document.getElementById('show-history-toggle').checked;
    // Force refresh
    updateScores();
}

// Cleanup intervals when page unloads
window.addEventListener('beforeunload', function() {
    if (statusInterval) clearInterval(statusInterval);
    if (logsInterval) clearInterval(logsInterval);
    if (livePlotInterval) clearInterval(livePlotInterval);
    if (scoresInterval) clearInterval(scoresInterval);
    if (reconInterval) clearInterval(reconInterval);
    if (selectionInterval) clearInterval(selectionInterval);
});

// --- Combined Reconstruction Display (Latest + Per-Symbol) ---
let reconInterval;
let symbolList = [];
let currentSymbolIndex = -1; // -1 means "Latest" is selected
let latestSample = null;

function initReconControls() {
    const prevBtn = document.getElementById('symbol-prev-btn');
    const nextBtn = document.getElementById('symbol-next-btn');
    const selector = document.getElementById('symbol-selector');
    
    if (prevBtn) prevBtn.addEventListener('click', () => selectPrevSymbol());
    if (nextBtn) nextBtn.addEventListener('click', () => selectNextSymbol());
    if (selector) selector.addEventListener('change', () => {
        const val = selector.value;
        if (val === '__latest__') {
            currentSymbolIndex = -1;
            showCurrentRecon();
        } else if (val) {
            const idx = symbolList.findIndex(e => String(e.symbol_id) === String(val));
            if (idx >= 0) {
                currentSymbolIndex = idx;
                showCurrentRecon();
            }
        }
    });
}

function startReconUpdates() {
    if (reconInterval) clearInterval(reconInterval);
    updateReconData();
    reconInterval = setInterval(updateReconData, 6000);
}

async function updateReconData() {
    try {
        // Fetch both latest reconstruction and per-symbol data
        const [latestResp, symbolsResp] = await Promise.all([
            fetch('/api/recon-sample'),
            fetch('/api/recon-symbols')
        ]);
        
        // Update latest sample
        if (latestResp.ok) {
            const latestData = await latestResp.json();
            latestSample = latestData.sample;
        }
        
        // Update symbol list
        if (symbolsResp.ok) {
            const symbolsData = await symbolsResp.json();
            const items = symbolsData.symbols || [];
            symbolList = items;
            
            const selector = document.getElementById('symbol-selector');
            const countEl = document.getElementById('symbol-count');
            
            if (selector) {
                const currentValue = selector.value;
                
                // Rebuild options
                selector.innerHTML = '<option value="__latest__">Latest</option>';
                items.forEach((entry) => {
                    const opt = document.createElement('option');
                    opt.value = entry.symbol_id;
                    opt.textContent = `Symbol ${entry.symbol_id}`;
                    selector.appendChild(opt);
                });
                
                // Restore selection if possible
                if (currentValue === '__latest__' || currentSymbolIndex === -1) {
                    selector.value = '__latest__';
                    currentSymbolIndex = -1;
                } else if (currentValue && items.find(e => String(e.symbol_id) === String(currentValue))) {
                    selector.value = currentValue;
                    currentSymbolIndex = items.findIndex(e => String(e.symbol_id) === String(currentValue));
                } else {
                    // Current selection no longer exists, default to Latest
                    selector.value = '__latest__';
                    currentSymbolIndex = -1;
                }
            }
            
            if (countEl) {
                const total = items.length + 1; // +1 for "Latest"
                countEl.textContent = `${total} view${total !== 1 ? 's' : ''}`;
            }
        }
        
        // Update display
        showCurrentRecon();
    } catch (e) {
        // silent
    }
}

function selectPrevSymbol() {
    // Navigation order: Latest (-1) -> Symbol 0 -> Symbol 1 -> ... -> Symbol N-1 -> Latest
    if (currentSymbolIndex === -1) {
        // From Latest, go to last symbol
        if (symbolList.length > 0) {
            currentSymbolIndex = symbolList.length - 1;
        }
    } else {
        currentSymbolIndex--;
        if (currentSymbolIndex < 0) {
            // Wrap to Latest
            currentSymbolIndex = -1;
        }
    }
    
    updateSelectorFromIndex();
    showCurrentRecon();
}

function selectNextSymbol() {
    // Navigation order: Latest (-1) -> Symbol 0 -> Symbol 1 -> ... -> Symbol N-1 -> Latest
    if (currentSymbolIndex === -1) {
        // From Latest, go to first symbol
        if (symbolList.length > 0) {
            currentSymbolIndex = 0;
        }
    } else {
        currentSymbolIndex++;
        if (currentSymbolIndex >= symbolList.length) {
            // Wrap back to Latest
            currentSymbolIndex = -1;
        }
    }
    
    updateSelectorFromIndex();
    showCurrentRecon();
}

function updateSelectorFromIndex() {
    const selector = document.getElementById('symbol-selector');
    if (!selector) return;
    
    if (currentSymbolIndex === -1) {
        selector.value = '__latest__';
    } else if (currentSymbolIndex >= 0 && currentSymbolIndex < symbolList.length) {
        selector.value = symbolList[currentSymbolIndex].symbol_id;
    }
}

function showCurrentRecon() {
    const container = document.getElementById('recon-display-container');
    if (!container) return;
    
    if (currentSymbolIndex === -1) {
        // Show latest
        renderReconSample(container, latestSample);
    } else if (currentSymbolIndex >= 0 && currentSymbolIndex < symbolList.length) {
        // Show selected symbol
        const entry = symbolList[currentSymbolIndex];
        renderReconSample(container, entry.sample);
    }
}

function renderReconSample(container, sample) {
    if (!sample) {
        container.innerHTML = '<div class="recon-empty">No reconstruction sample yet</div>';
        return;
    }
    const target = sample.target || [];
    const recon = sample.reconstruction || [];
    const dir = sample.direction || '';
    
    // Check if we have sequence data (multi-position messages)
    const hasSequence = sample.message_symbols_abs && Array.isArray(sample.message_symbols_abs);
    let symbolDisplay = '';
    
    if (hasSequence) {
        const seqLen = sample.sequence_length || sample.message_symbols_abs.length;
        const localSyms = sample.message_symbols_local || [];
        const absSyms = sample.message_symbols_abs || [];
        
        // Build display showing all positions
        const positions = [];
        for (let i = 0; i < seqLen; i++) {
            const loc = localSyms[i] != null ? localSyms[i] : '?';
            const abs = absSyms[i] != null ? absSyms[i] : '?';
            positions.push(`Pos ${i}: ${loc}/${abs}`);
        }
        symbolDisplay = `<span style="margin-left:8px;color:#666"><strong>Message</strong>: [${positions.join(', ')}]</span>`;
    } else {
        // Backward compatibility: single symbol
        const symLocal = (sample.message_symbol_local != null) ? sample.message_symbol_local : '-';
        const symAbs = (sample.message_symbol_abs != null) ? sample.message_symbol_abs : '-';
        symbolDisplay = `<span style="margin-left:8px;color:#666"><strong>Local/Abs</strong>: ${symLocal}/${symAbs}</span>`;
    }
    
    // Get sizes - use actual grid dimensions as fallback
    const targetSize = sample.target_size || [target.length, target[0]?.length || 0];
    const predSize = sample.predicted_size || [recon.length, recon[0]?.length || 0];
    const sizeMatch = targetSize[0] === predSize[0] && targetSize[1] === predSize[1];
    
    const targetGrid = gridToHtml(target);
    const reconGrid = gridToHtml(recon);
    
    const sizeClass = sizeMatch ? 'size-match' : 'size-mismatch';
    
    container.innerHTML = `
        <div class="recon-display ${sizeClass}">
            <div class="recon-header">
                <span><strong>Direction</strong>: ${dir}</span>
                ${symbolDisplay}
            </div>
            <div class="recon-grids">
                <div class="recon-grid-block">
                    <div class="recon-title">Target (${targetSize[0]}×${targetSize[1]})</div>
                    ${targetGrid}
                </div>
                <div class="recon-grid-block">
                    <div class="recon-title">Reconstruction (${predSize[0]}×${predSize[1]})</div>
                    ${reconGrid}
                </div>
            </div>
        </div>
    `;
}

function gridToHtml(grid) {
    if (!grid || !grid.length) return '<div class="grid grid-empty">No grid</div>';
    const h = grid.length;
    const w = grid[0].length || 0;
    const palette = [
        '#000000','#0074D9','#2ECC40','#FF4136','#FF851B',
        '#B10DC9','#7FDBFF','#01FF70','#AAAAAA','#FFDC00',
        '#001f3f','#39CCCC','#01FF70','#85144b','#F012BE',
        '#3D9970','#111111','#AAAAAA','#FF851B','#DDDDDD'
    ];
    let html = '<div class="grid" style="display:inline-grid; grid-template-columns: '+('1.2em '.repeat(w)).trim()+'; gap: 2px;">';
    for (let y=0; y<h; y++) {
        for (let x=0; x<w; x++) {
            const v = grid[y][x] || 0;
            const color = palette[v % palette.length];
            html += '<div class="cell" style="width:1.2em;height:1.2em;background:'+color+'"></div>';
        }
    }
    html += '</div>';
    return html;
}

// --- Selection Task Display (Input → Output) ---
let selectionInterval;

function startSelectionUpdates() {
    if (selectionInterval) clearInterval(selectionInterval);
    updateSelectionSample();
    selectionInterval = setInterval(updateSelectionSample, 8000);
}

async function updateSelectionSample() {
    try {
        const container = document.getElementById('selection-display-container');
        if (!container) return;
        
        const resp = await fetch('/api/selection-sample');
        if (!resp.ok) return;
        
        const data = await resp.json();
        renderSelectionSample(container, data.sample);
    } catch (e) {
        // silent
    }
}

function renderSelectionSample(container, sample) {
    if (!sample) {
        container.innerHTML = '<div class="recon-empty">No selection sample yet</div>';
        return;
    }
    
    const inputPuzzle = sample.input_puzzle || [];
    const outputPuzzle = sample.output_puzzle || [];
    const dir = sample.direction || '';
    const correct = sample.selection_correct;
    const confidence = sample.confidence || 0;
    
    // Check if we have sequence data (multi-position messages)
    const hasSequence = sample.message_symbols_abs && Array.isArray(sample.message_symbols_abs);
    let symbolDisplay = '';
    
    if (hasSequence) {
        const seqLen = sample.sequence_length || sample.message_symbols_abs.length;
        const localSyms = sample.message_symbols_local || [];
        const absSyms = sample.message_symbols_abs || [];
        
        // Build display showing all positions
        const positions = [];
        for (let i = 0; i < seqLen; i++) {
            const loc = localSyms[i] != null ? localSyms[i] : '?';
            const abs = absSyms[i] != null ? absSyms[i] : '?';
            positions.push(`Pos ${i}: ${loc}/${abs}`);
        }
        symbolDisplay = `<span style="margin-left:8px;color:#666"><strong>Message</strong>: [${positions.join(', ')}]</span>`;
    } else {
        // Backward compatibility: single symbol
        const symLocal = (sample.message_symbol_local != null) ? sample.message_symbol_local : '-';
        const symAbs = (sample.message_symbol_abs != null) ? sample.message_symbol_abs : '-';
        symbolDisplay = `<span style="margin-left:8px;color:#666"><strong>Local/Abs</strong>: ${symLocal}/${symAbs}</span>`;
    }
    
    // Get sizes
    const inputSize = sample.input_size || [inputPuzzle.length, inputPuzzle[0]?.length || 0];
    const outputSize = sample.output_size || [outputPuzzle.length, outputPuzzle[0]?.length || 0];
    
    const inputGrid = gridToHtml(inputPuzzle);
    const outputGrid = gridToHtml(outputPuzzle);
    
    const correctClass = correct ? 'selection-correct' : 'selection-incorrect';
    const correctIndicator = correct ? '✓ Correct' : '✗ Incorrect';
    const correctColor = correct ? 'green' : 'red';
    
    container.innerHTML = `
        <div class="recon-display ${correctClass}">
            <div class="recon-header">
                <span><strong>Direction</strong>: ${dir}</span>
                ${symbolDisplay}
                <span style="margin-left:16px;color:${correctColor};font-weight:bold">${correctIndicator}</span>
                <span style="margin-left:8px;color:#666"><strong>Confidence</strong>: ${(confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="recon-grids">
                <div class="recon-grid-block">
                    <div class="recon-title">Input Puzzle (${inputSize[0]}×${inputSize[1]})</div>
                    ${inputGrid}
                </div>
                <div class="recon-grid-block">
                    <div class="recon-title">Output Puzzle (${outputSize[0]}×${outputSize[1]})</div>
                    ${outputGrid}
                </div>
            </div>
        </div>
    `;
}