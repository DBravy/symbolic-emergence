// Global variables
let currentConfig = {};
let autoScroll = true;
let statusInterval;
let logsInterval;

// Live plot auto-refresh
let livePlotInterval;
let scoresInterval;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadConfig();
    startStatusUpdates();
    startLogUpdates();
    // Keep manual refresh around but start live updates automatically
    startLivePlotUpdates();
    startScoresUpdates();
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
}

async function saveConfig() {
    // Collect form data
    const config = {};
    const formElements = [
        'max_global_phases', 'initial_puzzle_count', 'training_cycles', 
        'puzzles_per_addition', 'learning_rate', 'num_distractors',
        'distractor_strategy', 'phase_change_indicator', 'embedding_dim',
        'hidden_dim', 'num_symbols', 'puzzle_symbols'
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

function resetConfig() {
    if (confirm('Reset all configuration to defaults?')) {
        // Reset to default values
        const defaults = {
            max_global_phases: 100,
            initial_puzzle_count: 4,
            training_cycles: 25,
            puzzles_per_addition: 3,
            learning_rate: 0.0000007,
            num_distractors: 3,
            distractor_strategy: 'random',
            phase_change_indicator: 'ges',
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
        const response = await fetch('/api/start', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            showNotification('Training started successfully', 'success');
            
            // Update UI to show running state
            document.getElementById('training-status').textContent = 'Running';
            document.getElementById('training-status').classList.add('running');
        } else {
            showNotification(result.message || 'Error starting training', 'error');
        }
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Error starting training', 'error');
    }
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
            
            // Update progress bar
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            const progress = info.progress || 0;
            
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
        }
        
        document.getElementById('process-pid').textContent = status.pid || 'None';
        
    } catch (error) {
        console.error('Error updating status:', error);
        document.getElementById('connection-status').textContent = 'Connection Error';
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

// Cleanup intervals when page unloads
window.addEventListener('beforeunload', function() {
    if (statusInterval) clearInterval(statusInterval);
    if (logsInterval) clearInterval(logsInterval);
    if (livePlotInterval) clearInterval(livePlotInterval);
    if (scoresInterval) clearInterval(scoresInterval);
});