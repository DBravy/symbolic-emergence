from flask import Flask, render_template, request, jsonify, send_file, after_this_request
import subprocess
import json
import os
import time
import psutil
import signal
from threading import Thread
import glob
import sys
from datetime import datetime
import fcntl
import select
import re
from collections import deque
import tempfile
import textwrap
import torch

# Use non-interactive backend for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    matplotlib = None
    plt = None
    PdfPages = None

app = Flask(__name__)

# Global variables to track training
current_process = None
current_config = {}
training_log = []
debug_info = []

# Phase management
PHASES_FILE = 'training_phases.json'
DEFAULT_PHASES = [
    {'id': 0, 'name': 'Communication Protocol', 'type': 'base', 'route': 'phase_0'},
    {'id': 'arc', 'name': 'ARC Solving', 'type': 'final', 'route': 'arc'}
]
current_phases = []

# Default configuration
DEFAULT_CONFIG = {
    'max_global_phases': 100,
    'first_pretrain_epochs': 100,
    'pretrain_epochs': 100,
    'initial_puzzle_count': 4,
    'initial_comm_symbols': 4,
    'first_training_cycles': 50,
    'training_cycles': 25,
    'early_stop_min_cycles': 5,
    'consolidation_tests': 5,
    'consolidation_threshold': 0.3,
    'puzzles_per_addition': 3,
    'repetitions_per_puzzle': 1,
    'num_distractors': 3,
    'distractor_strategy': 'random',
    'phase_change_indicator': 'ges',
    'learning_rate': 7e-7,
    'embedding_dim': 512,
    'hidden_dim': 1024,
    'num_symbols': 100,
    'puzzle_symbols': 10,
    'max_seq_length': 10,
    'current_seq_length': 1,
    'output_dir': './outputs',
    # NEW: Optional human-readable title for this training run
    'run_title': ''
}

# Runtime training mode control file
CONTROL_FILE = 'training_control.json'
SNAPSHOT_DIR = os.path.join(DEFAULT_CONFIG.get('output_dir', './outputs'), 'snapshots')

def log_debug(message):
    """Add debug message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    debug_entry = {
        'timestamp': timestamp,
        'message': str(message)
    }
    debug_info.append(debug_entry)
    # print(f"[DEBUG {timestamp}] {message}")
    
    # Keep only last 500 debug entries
    if len(debug_info) > 500:
        debug_info.pop(0)

def load_phases():
    """Load phases from file or return defaults"""
    global current_phases
    if os.path.exists(PHASES_FILE):
        try:
            with open(PHASES_FILE, 'r') as f:
                current_phases = json.load(f)
            log_debug(f"Loaded {len(current_phases)} phases from file")
        except Exception as e:
            log_debug(f"Error loading phases: {e}, using defaults")
            current_phases = DEFAULT_PHASES.copy()
    else:
        current_phases = DEFAULT_PHASES.copy()
    return current_phases

def save_phases(phases):
    """Save phases to file"""
    global current_phases
    current_phases = phases
    try:
        with open(PHASES_FILE, 'w') as f:
            json.dump(phases, f, indent=2)
        log_debug(f"Saved {len(phases)} phases to file")
        return True
    except Exception as e:
        log_debug(f"Error saving phases: {e}")
        return False

@app.route('/')
def index():
    phases = load_phases()
    return render_template('index.html', phases=phases)

@app.route('/phase/<phase_id>')
def phase_page(phase_id):
    phases = load_phases()
    # Find the phase
    phase = None
    phase_index = None
    for i, p in enumerate(phases):
        if str(p['id']) == str(phase_id):
            phase = p
            phase_index = i
            break
    
    if not phase:
        return "Phase not found", 404
    
    # Route to appropriate template
    if phase['type'] == 'base' and phase['id'] == 0:
        return render_template('index.html', phases=phases)
    elif phase['type'] == 'final':
        return render_template('arc.html', phases=phases)
    else:
        # Intermediate phase - use placeholder template
        return render_template('phase_intermediate.html', phase=phase, phases=phases, phase_index=phase_index)

@app.route('/arc')
def arc_page():
    phases = load_phases()
    return render_template('arc.html', phases=phases)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    global current_config
    if not current_config:
        current_config = DEFAULT_CONFIG.copy()
    return jsonify(current_config)

@app.route('/api/config', methods=['POST'])
def set_config():
    """Update current configuration and persist to training_config.json"""
    global current_config
    try:
        data = request.get_json(force=True)
        if not current_config:
            current_config = DEFAULT_CONFIG.copy()
        # Shallow update only for known keys
        for key, value in data.items():
            current_config[key] = value
        with open('training_config.json', 'w') as f:
            json.dump(current_config, f, indent=2)
        log_debug("Configuration updated via API")
        return jsonify(current_config)
    except Exception as e:
        log_debug(f"Error updating config: {e}")
        return jsonify({ 'error': str(e) }), 400

@app.route('/api/debug', methods=['GET'])
def debug_status():
    """Get detailed debug information"""
    log_debug("Debug status requested")
    
    debug_data = {
        'current_working_directory': os.getcwd(),
        'python_executable': sys.executable,
        'environment_path': os.environ.get('PATH', ''),
        'process_info': {
            'running': current_process is not None and current_process.poll() is None,
            'pid': current_process.pid if current_process else None,
            'poll_result': current_process.poll() if current_process else None
        },
        'file_checks': {
            'training_script_paths': {
                'src/train_selection.py': {
                    'exists': os.path.exists('src/train_selection.py'),
                    'size': os.path.getsize('src/train_selection.py') if os.path.exists('src/train_selection.py') else 0,
                    'readable': os.access('src/train_selection.py', os.R_OK) if os.path.exists('src/train_selection.py') else False
                },
                'train_selection.py': {
                    'exists': os.path.exists('train_selection.py'),
                    'size': os.path.getsize('train_selection.py') if os.path.exists('train_selection.py') else 0,
                    'readable': os.access('train_selection.py', os.R_OK) if os.path.exists('train_selection.py') else False
                }
            },
            'arc_data_paths': {
                'src/arc-agi_test_challenges.json': {
                    'exists': os.path.exists('src/arc-agi_test_challenges.json'),
                    'size': os.path.getsize('src/arc-agi_test_challenges.json') if os.path.exists('src/arc-agi_test_challenges.json') else 0
                },
                'arc-agi_test_challenges.json': {
                    'exists': os.path.exists('arc-agi_test_challenges.json'),
                    'size': os.path.getsize('arc-agi_test_challenges.json') if os.path.exists('arc-agi_test_challenges.json') else 0
                }
            },
            'config_file': {
                'exists': os.path.exists('training_config.json'),
                'size': os.path.getsize('training_config.json') if os.path.exists('training_config.json') else 0
            },
            'status_file': {
                'exists': os.path.exists('training_status.json'),
                'size': os.path.getsize('training_status.json') if os.path.exists('training_status.json') else 0
            },
            'output_dir': current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir')) if current_config else DEFAULT_CONFIG.get('output_dir')
        },
        'recent_logs': training_log[-20:] if training_log else [],
        'debug_messages': debug_info[-50:] if debug_info else []
    }
    
    # Try to read status file
    if os.path.exists('training_status.json'):
        try:
            with open('training_status.json', 'r') as f:
                debug_data['status_file_content'] = json.load(f)
        except Exception as e:
            debug_data['status_file_error'] = str(e)
    
    # Check if we can run basic Python commands
    try:
        result = subprocess.run([sys.executable, '-c', 'print("test")'], 
                              capture_output=True, text=True, timeout=5)
        debug_data['python_test'] = {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        debug_data['python_test'] = {'error': str(e)}
    
    return jsonify(debug_data)

@app.route('/api/test-training', methods=['POST'])
def test_training():
    """Test if training script can be started (without actually training)"""
    log_debug("Testing training script startup")
    
    # Find training script
    script_path = None
    if os.path.exists('src/train_selection.py'):
        script_path = 'src/train_selection.py'
    elif os.path.exists('train_selection.py'):
        script_path = 'train_selection.py'
    else:
        return jsonify({
            'success': False,
            'error': 'No training script found',
            'checked_paths': ['src/train_selection.py', 'train_selection.py']
        })
    
    log_debug(f"Found training script at: {script_path}")
    
    # Test basic import
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            f'import sys; sys.path.insert(0, "{os.path.dirname(script_path) or "."}"); '
            f'import importlib.util; '
            f'spec = importlib.util.spec_from_file_location("test_module", "{script_path}"); '
            f'module = importlib.util.module_from_spec(spec); '
            f'print("Import successful")'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            log_debug(f"Import test failed: {result.stderr}")
            return jsonify({
                'success': False,
                'error': 'Script import failed',
                'details': {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            })
        
        log_debug("Import test successful")
        
    except Exception as e:
        log_debug(f"Import test error: {e}")
        return jsonify({
            'success': False,
            'error': f'Import test error: {str(e)}'
        })
    
    # Test help/args parsing
    try:
        result = subprocess.run([
            sys.executable, script_path, '--help'
        ], capture_output=True, text=True, timeout=10)
        
        log_debug(f"Help test result: returncode={result.returncode}")
        
    except Exception as e:
        log_debug(f"Help test error: {e}")
    
    return jsonify({
        'success': True,
        'script_path': script_path,
        'message': 'Training script appears to be working'
    })

@app.route('/api/start', methods=['POST'])
def start_training():
    """Start training process with enhanced debugging"""
    global current_process, training_log, current_config
    
    log_debug("=== STARTING TRAINING PROCESS ===")
    
    # Check if already running
    if current_process and current_process.poll() is None:
        log_debug("Training already running")
        return jsonify({'status': 'error', 'message': 'Training already running'})
    
    try:
        # Clear previous logs
        training_log = []
        log_debug("Cleared previous logs")
        
        # Ensure config file exists
        if not current_config:
            current_config = DEFAULT_CONFIG.copy()
            log_debug("Using default config")
        
        # Allow overrides from request body
        data = None
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        if data and isinstance(data, dict):
            # Optional override of current_seq_length or other config keys
            for k in ['current_seq_length', 'max_seq_length', 'embedding_dim', 'hidden_dim', 'num_symbols', 'puzzle_symbols', 'learning_rate', 'run_title']:
                if k in data:
                    current_config[k] = data[k]
        with open('training_config.json', 'w') as f:
            json.dump(current_config, f, indent=2)
        log_debug("Config file written")
        # Ensure output_dir exists
        try:
            output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs'))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                log_debug(f"Ensured output_dir exists: {output_dir}")
        except Exception as e:
            log_debug(f"Could not create output_dir: {e}")
        
        # Clear previous score logs to reset history between runs
        try:
            for fname in ['unseen_testing_log.txt', 'novel_symbol_unseen_testing_log.txt', 'phase_training_log.txt']:
                # Try clearing in output_dir and cwd
                candidates = []
                if output_dir:
                    candidates.append(os.path.join(output_dir, fname))
                candidates.append(fname)
                for p in candidates:
                    if os.path.exists(p):
                        with open(p, 'w') as f:
                            f.write('')
                        log_debug(f"Cleared log file at start: {p}")
        except Exception as e:
            log_debug(f"Error clearing score logs: {e}")
        
        # Check if the training script exists
        training_script = 'src/train_selection.py'
        if not os.path.exists(training_script):
            training_script = 'train_selection.py'
            if not os.path.exists(training_script):
                error_msg = f'Training script not found at {training_script} or src/train_selection.py'
                log_debug(error_msg)
                return jsonify({'status': 'error', 'message': error_msg})
        
        log_debug(f"Using training script: {training_script}")
        
        # Check if ARC data file exists
        arc_file = 'src/arc-agi_test_challenges.json'
        if not os.path.exists(arc_file):
            arc_file = 'arc-agi_test_challenges.json'
            if not os.path.exists(arc_file):
                error_msg = f'ARC dataset file not found'
                log_debug(error_msg)
                return jsonify({'status': 'error', 'message': error_msg})
        
        log_debug(f"Using ARC file: {arc_file}")
        
        # Prepare command
        cmd = [
            sys.executable, training_script,
            '--config', 'training_config.json',
            '--status-file', 'training_status.json',
            '--web-mode',
            '--control-file', CONTROL_FILE
        ]

        # Optional: resume-from snapshot (explicit or selected)
        resume_path = None
        if data and isinstance(data, dict):
            # Direct path or filename
            candidate = data.get('resume_from') or data.get('resume_filename')
            if candidate:
                # Resolve relative filename within snapshots dir
                out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs'))
                snap_dir = os.path.join(out_dir, 'snapshots')
                if os.path.isabs(candidate) and os.path.exists(candidate):
                    resume_path = candidate
                else:
                    cand_path = os.path.join(snap_dir, os.path.basename(candidate))
                    if os.path.exists(cand_path):
                        resume_path = cand_path
        # If still not provided, use selected_snapshot.json if present
        if not resume_path:
            try:
                out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs'))
                snap_dir = os.path.join(out_dir, 'snapshots')
                sel = os.path.join(snap_dir, 'selected_snapshot.json')
                if os.path.exists(sel):
                    with open(sel, 'r') as f:
                        sel_data = json.load(f) or {}
                    fname = sel_data.get('filename')
                    if fname:
                        cand = os.path.join(snap_dir, os.path.basename(fname))
                        if os.path.exists(cand):
                            resume_path = cand
            except Exception as e:
                log_debug(f"Error reading selected snapshot: {e}")

        if resume_path:
            cmd.extend(['--resume-from', resume_path])

        # Optional: freeze positions for progressive training
        if data and isinstance(data, dict) and data.get('freeze_positions'):
            try:
                fp = data.get('freeze_positions')
                if isinstance(fp, list) and all(isinstance(x, int) for x in fp):
                    arg = ','.join(str(x) for x in fp)
                    cmd.extend(['--freeze-positions', arg])
            except Exception:
                pass
        
        log_debug(f"Command: {' '.join(cmd)}")
        log_debug(f"Working directory: {os.getcwd()}")
        
        # Start training process with enhanced settings
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            universal_newlines=True,
            env=os.environ.copy()  # Ensure environment is passed
        )
        
        log_debug(f"Process started with PID: {current_process.pid}")
        
        # Make stdout and stderr non-blocking on Unix-like systems
        if hasattr(fcntl, 'fcntl'):
            try:
                fd_stdout = current_process.stdout.fileno()
                fd_stderr = current_process.stderr.fileno()
                
                flags_stdout = fcntl.fcntl(fd_stdout, fcntl.F_GETFL)
                flags_stderr = fcntl.fcntl(fd_stderr, fcntl.F_GETFL)
                
                fcntl.fcntl(fd_stdout, fcntl.F_SETFL, flags_stdout | os.O_NONBLOCK)
                fcntl.fcntl(fd_stderr, fcntl.F_SETFL, flags_stderr | os.O_NONBLOCK)
                
                log_debug("Set non-blocking I/O")
            except Exception as e:
                log_debug(f"Could not set non-blocking I/O: {e}")
        
        # Start enhanced log monitoring thread
        log_thread = Thread(target=enhanced_monitor_logs, daemon=True)
        log_thread.start()
        log_debug("Started log monitoring thread")
        
        # Wait a moment to see if process starts successfully
        time.sleep(2)
        poll_result = current_process.poll()
        
        if poll_result is not None:
            # Process already terminated, capture error
            stdout, stderr = current_process.communicate()
            error_msg = f"Training script failed immediately (code {poll_result}).\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            log_debug(error_msg)
            training_log.append({
                'timestamp': time.time(),
                'type': 'error',
                'message': error_msg
            })
            current_process = None
            return jsonify({'status': 'error', 'message': error_msg})
        
        log_debug("Process appears to be running successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'Training started (web mode - enhanced monitoring)',
            'pid': current_process.pid,
            'script_path': training_script
        })
        
    except Exception as e:
        error_msg = f"Failed to start training: {str(e)}"
        log_debug(error_msg)
        training_log.append({
            'timestamp': time.time(),
            'type': 'error',
            'message': error_msg
        })
        return jsonify({'status': 'error', 'message': error_msg})

@app.route('/api/training-mode', methods=['GET', 'POST'])
def training_mode():
    """Get or set runtime training mode (selection | reconstruction)."""
    try:
        if request.method == 'POST':
            data = request.get_json(force=True) or {}
            mode = data.get('mode', '').strip().lower()
            if mode not in ('selection', 'reconstruction'):
                return jsonify({'error': 'mode must be selection or reconstruction'}), 400
            # Write control file atomically
            tmp = CONTROL_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'mode': mode}, f)
            os.replace(tmp, CONTROL_FILE)
            log_debug(f"Training mode set to: {mode}")
            return jsonify({'mode': mode, 'status': 'ok'})
        # GET: read current mode
        if os.path.exists(CONTROL_FILE):
            with open(CONTROL_FILE, 'r') as f:
                data = json.load(f)
            mode = data.get('mode', 'selection')
        else:
            mode = 'selection'
        return jsonify({'mode': mode})
    except Exception as e:
        log_debug(f"Error handling training mode: {e}")
        return jsonify({'error': str(e)}), 500

def enhanced_monitor_logs():
    """Enhanced log monitoring with better error handling and debugging"""
    global current_process, training_log
    
    log_debug("Starting enhanced log monitoring")
    
    if not current_process:
        log_debug("No process to monitor")
        return
    
    last_activity = time.time()
    status_file_last_check = 0
    
    try:
        while current_process and current_process.poll() is None:
            activity_this_cycle = False
            
            # Check status file for updates
            current_time = time.time()
            if current_time - status_file_last_check > 1:  # Check every second
                if os.path.exists('training_status.json'):
                    try:
                        stat_info = os.stat('training_status.json')
                        if stat_info.st_mtime > status_file_last_check:
                            with open('training_status.json', 'r') as f:
                                status = json.load(f)
                            log_debug(f"Status file updated: {status}")
                            training_log.append({
                                'timestamp': time.time(),
                                'type': 'status_update',
                                'message': f"Status: {status.get('status', 'unknown')} - {status.get('message', 'no message')}"
                            })
                            activity_this_cycle = True
                    except Exception as e:
                        log_debug(f"Error reading status file: {e}")
                
                status_file_last_check = current_time
            
            # Read stdout and stderr using select (Unix) or polling (Windows)
            if hasattr(select, 'select'):
                # Unix-like systems
                try:
                    ready, _, _ = select.select([current_process.stdout, current_process.stderr], [], [], 0.1)
                    
                    for stream in ready:
                        try:
                            if stream == current_process.stdout:
                                line = stream.readline()
                                if line:
                                    line = line.strip()
                                    log_debug(f"STDOUT: {line}")
                                    training_log.append({
                                        'timestamp': time.time(),
                                        'type': 'stdout',
                                        'message': line
                                    })
                                    activity_this_cycle = True
                            elif stream == current_process.stderr:
                                line = stream.readline()
                                if line:
                                    line = line.strip()
                                    log_debug(f"STDERR: {line}")
                                    training_log.append({
                                        'timestamp': time.time(),
                                        'type': 'stderr',
                                        'message': line
                                    })
                                    activity_this_cycle = True
                        except Exception as e:
                            log_debug(f"Error reading from stream: {e}")
                
                except Exception as e:
                    log_debug(f"Error in select: {e}")
            else:
                # Windows - use polling approach
                try:
                    # Try to read stdout
                    while True:
                        try:
                            line = current_process.stdout.readline()
                            if not line:
                                break
                            line = line.strip()
                            if line:
                                log_debug(f"STDOUT: {line}")
                                training_log.append({
                                    'timestamp': time.time(),
                                    'type': 'stdout',
                                    'message': line
                                })
                                activity_this_cycle = True
                        except:
                            break
                    
                    # Try to read stderr
                    while True:
                        try:
                            line = current_process.stderr.readline()
                            if not line:
                                break
                            line = line.strip()
                            if line:
                                log_debug(f"STDERR: {line}")
                                training_log.append({
                                    'timestamp': time.time(),
                                    'type': 'stderr',
                                    'message': line
                                })
                                activity_this_cycle = True
                        except:
                            break
                            
                except Exception as e:
                    log_debug(f"Error reading output (Windows): {e}")
            
            if activity_this_cycle:
                last_activity = time.time()
            
            # Check for hanging process
            if time.time() - last_activity > 30:  # No activity for 30 seconds
                log_debug("Warning: No process activity for 30 seconds")
                training_log.append({
                    'timestamp': time.time(),
                    'type': 'warning',
                    'message': 'No process activity for 30 seconds - process may be hanging'
                })
                last_activity = time.time()  # Reset to avoid spam
            
            # Keep log size manageable
            if len(training_log) > 1000:
                training_log = training_log[-800:]  # Keep last 800 entries
            
            time.sleep(0.1)
        
        # Process finished
        if current_process:
            poll_result = current_process.poll()
            log_debug(f"Process finished with return code: {poll_result}")
            
            try:
                stdout, stderr = current_process.communicate(timeout=5)
                if stdout:
                    log_debug(f"Final STDOUT: {stdout}")
                    training_log.append({
                        'timestamp': time.time(),
                        'type': 'final_stdout',
                        'message': stdout
                    })
                if stderr:
                    log_debug(f"Final STDERR: {stderr}")
                    training_log.append({
                        'timestamp': time.time(),
                        'type': 'final_stderr',
                        'message': stderr
                    })
            except subprocess.TimeoutExpired:
                log_debug("Timeout getting final output")
                
    except Exception as e:
        error_msg = f'Log monitoring error: {str(e)}'
        log_debug(error_msg)
        training_log.append({
            'timestamp': time.time(),
            'type': 'monitor_error',
            'message': error_msg
        })

@app.route('/api/stop', methods=['POST'])
def stop_training():
    """Stop training process"""
    global current_process
    
    log_debug("Stop training requested")
    
    if not current_process:
        return jsonify({'status': 'error', 'message': 'No training process running'})
    
    try:
        log_debug(f"Terminating process {current_process.pid}")
        current_process.terminate()
        
        # Wait a bit, then force kill if necessary
        try:
            current_process.wait(timeout=5)
            log_debug("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            log_debug("Force killing process")
            current_process.kill()
        
        current_process = None
        return jsonify({'status': 'success', 'message': 'Training stopped'})
        
    except Exception as e:
        error_msg = f"Error stopping training: {str(e)}"
        log_debug(error_msg)
        return jsonify({'status': 'error', 'message': error_msg})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get training status"""
    global current_process
    
    # Check process status
    is_running = current_process is not None and current_process.poll() is None
    
    # Read status file if it exists
    status_info = {'status': 'idle', 'progress': 0, 'message': 'No training running'}
    if os.path.exists('training_status.json'):
        try:
            with open('training_status.json', 'r') as f:
                status_info = json.load(f)
        except Exception as e:
            status_info['file_error'] = str(e)
    
    return jsonify({
        'running': is_running,
        'pid': current_process.pid if is_running else None,
        'poll_result': current_process.poll() if current_process else None,
        'status_info': status_info
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent training logs"""
    global training_log
    return jsonify({'logs': training_log[-200:]})  # Last 200 lines

@app.route('/api/recon-sample', methods=['GET'])
def get_recon_sample():
    """Return the most recent reconstruction sample recorded in the status/log stream if available."""
    try:
        # Try to read from training_status.json first for structured data (future-proof)
        sample = None
        if os.path.exists('training_status.json'):
            try:
                with open('training_status.json', 'r') as f:
                    st = json.load(f)
                sample = st.get('last_recon_sample')
            except Exception:
                sample = None
        # Fallback: scan recent logs for an embedded recon sample entry
        if sample is None:
            for entry in reversed(training_log[-300:]):
                msg = entry.get('message', '') if isinstance(entry, dict) else ''
                if 'RECON_SAMPLE_JSON:' in msg:
                    try:
                        payload = msg.split('RECON_SAMPLE_JSON:', 1)[1].strip()
                        sample = json.loads(payload)
                        break
                    except Exception:
                        continue
        return jsonify({'sample': sample})
    except Exception as e:
        log_debug(f"Error retrieving recon sample: {e}")
        return jsonify({'error': str(e)}), 500


# --- NEW: Latest reconstruction per symbol ---
@app.route('/api/recon-symbols', methods=['GET'])
def get_recon_symbols():
    """Return the latest reconstruction sample per symbol as recorded in the status file."""
    try:
        symbols = []
        # Prefer structured status file
        if os.path.exists('training_status.json'):
            try:
                with open('training_status.json', 'r') as f:
                    st = json.load(f) or {}
                mapping = st.get('symbol_recon_samples') or {}
                # Normalize into array of {symbol_id, sample}
                for k, v in mapping.items():
                    try:
                        sid = int(k)
                    except Exception:
                        # ignore non-integer keys
                        continue
                    symbols.append({'symbol_id': sid, 'sample': v})
                # Sort by symbol id
                symbols.sort(key=lambda x: x['symbol_id'])
            except Exception as e:
                log_debug(f"Error reading recon symbols from status file: {e}")
        return jsonify({'symbols': symbols})
    except Exception as e:
        log_debug(f"Error retrieving recon symbols: {e}")
        return jsonify({'symbols': []})

@app.route('/api/debug-logs', methods=['GET'])
def get_debug_logs():
    """Get debug messages"""
    global debug_info
    return jsonify({'debug_logs': debug_info[-100:]})

@app.route('/api/plots', methods=['GET'])
def get_plots():
    """Get list of available plots"""
    plot_files = []
    patterns = ['*.png', '*.jpg', '*.jpeg']
    
    # Prefer output_dir if set
    output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
    try:
        if output_dir and os.path.isdir(output_dir):
            for pattern in patterns:
                plot_files.extend([os.path.join(output_dir, os.path.basename(p)) for p in glob.glob(os.path.join(output_dir, pattern))])
    except Exception as e:
        log_debug(f"Error scanning output_dir for plots: {e}")
    
    # Also scan cwd as fallback
    for pattern in patterns:
        plot_files.extend(glob.glob(pattern))
    
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in plot_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return jsonify({'plots': unique_files})

@app.route('/api/plots/latest', methods=['GET'])
def get_latest_plot():
    """Return the latest plot filename based on modification time"""
    patterns = ['*.png', '*.jpg', '*.jpeg']
    output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
    candidates = []
    try:
        if output_dir and os.path.isdir(output_dir):
            for pattern in patterns:
                candidates.extend(glob.glob(os.path.join(output_dir, pattern)))
    except Exception as e:
        log_debug(f"Error scanning output_dir for latest plot: {e}")
    # Also include cwd as fallback
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return jsonify({'latest': None})
    # Pick newest by mtime
    try:
        latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
        return jsonify({'latest': latest_path, 'mtime': os.path.getmtime(latest_path)})
    except Exception as e:
        log_debug(f"Error selecting latest plot: {e}")
        return jsonify({'latest': None})


def _find_latest_plot_path():
    """Return latest plot file path or None."""
    patterns = ['*.png', '*.jpg', '*.jpeg']
    output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
    candidates = []
    try:
        if output_dir and os.path.isdir(output_dir):
            for pattern in patterns:
                candidates.extend(glob.glob(os.path.join(output_dir, pattern)))
    except Exception as e:
        log_debug(f"Error scanning output_dir for latest plot: {e}")
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    try:
        latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
        return latest_path
    except Exception as e:
        log_debug(f"Error selecting latest plot: {e}")
        return None

# --- restore plot serving route ---
@app.route('/api/plot/<path:filename>')
def serve_plot(filename):
    """Serve plot file from output_dir or cwd"""
    # Security: normalize path to avoid directory traversal
    safe_name = os.path.basename(filename)

    # Determine output directory
    output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
    candidate_paths = []
    if output_dir:
        candidate_paths.append(os.path.join(output_dir, safe_name))
    candidate_paths.append(safe_name)

    for path in candidate_paths:
        if os.path.exists(path) and path.endswith(('.png', '.jpg', '.jpeg')):
            try:
                return send_file(path)
            except Exception as e:
                log_debug(f"Error sending file {path}: {e}")
                return jsonify({'error': 'Error reading file'}), 500
    return jsonify({'error': 'File not found'}), 404

# --- NEW: Report generation API ---
@app.route('/api/report', methods=['GET'])
def generate_report():
    """Generate a PDF report with latest plot, configuration, and score history."""
    # Check matplotlib availability
    if PdfPages is None or plt is None:
        return jsonify({'error': 'Matplotlib is not available on the server to generate PDF reports.'}), 500

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    title = (cfg.get('run_title') or '').strip()
    safe_title = re.sub(r'[^A-Za-z0-9._-]+', '_', title) if title else ''
    report_filename = f'{safe_title + "_" if safe_title else ""}run_report_{timestamp}.pdf'

    # Gather data
    cfg = current_config.copy() if current_config else DEFAULT_CONFIG.copy()

    # Scores using existing helpers
    output_dir = cfg.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs'))
    unseen_candidates = [
        os.path.join(output_dir, 'unseen_testing_log.txt') if output_dir else None,
        'unseen_testing_log.txt',
        os.path.join('src', 'unseen_testing_log.txt')
    ]
    novel_candidates = [
        os.path.join(output_dir, 'novel_symbol_unseen_testing_log.txt') if output_dir else None,
        'novel_symbol_unseen_testing_log.txt',
        os.path.join('src', 'novel_symbol_unseen_testing_log.txt')
    ]
    unseen_lines, unseen_path = _read_tail_lines(unseen_candidates)
    novel_lines, novel_path = _read_tail_lines(novel_candidates)

    unseen_latest = _parse_test_summary_from_lines(unseen_lines, 'Unseen testing summary') if unseen_lines is not None else None
    novel_latest = _parse_test_summary_from_lines(novel_lines, 'Novel symbol induction summary') if novel_lines is not None else None

    unseen_history = []
    novel_history = []
    if unseen_path:
        all_unseen = _read_all_lines(unseen_path)
        if all_unseen is not None:
            unseen_history = _parse_all_summaries_from_lines(all_unseen, 'Unseen testing summary')
    if novel_path:
        all_novel = _read_all_lines(novel_path)
        if all_novel is not None:
            novel_history = _parse_all_summaries_from_lines(all_novel, 'Novel symbol induction summary')

    latest_plot_path = _find_latest_plot_path()

    # Create PDF
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp_path = tmp.name
        tmp.close()

        with PdfPages(tmp_path) as pdf:
            # Page 1: Title and configuration summary
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
            fig.suptitle(title if title else 'Run Report', fontsize=18, y=0.98)
            ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
            ax.axis('off')

            lines = []
            if title:
                lines.append(f'Title: {title}')
            lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            lines.append('')
            lines.append('Configuration:')
            # Nicely format configuration
            for key in sorted(cfg.keys()):
                val = cfg[key]
                line = f'  - {key}: {val}'
                wrapped = textwrap.wrap(line, width=90)
                lines.extend(wrapped)
            lines.append('')
            if unseen_latest:
                lines.append('Latest Unseen Test:')
                lines.append(f"  - Overall: {unseen_latest.get('correct','-')}/{unseen_latest.get('num_tests','-')} (acc={(unseen_latest.get('accuracy',0)*100):.1f}%)")
                if 'a1_to_a2_accuracy' in unseen_latest:
                    lines.append(f"  - A1→A2: {(unseen_latest.get('a1_to_a2_accuracy',0)*100):.1f}%  |  A2→A1: {(unseen_latest.get('a2_to_a1_accuracy',0)*100):.1f}%")
                if 'ges1_ma' in unseen_latest or 'ges2_ma' in unseen_latest:
                    lines.append(f"  - GES (MA): A1={unseen_latest.get('ges1_ma','-')}, A2={unseen_latest.get('ges2_ma','-')}")
                lines.append('')
            if novel_latest:
                lines.append('Latest Novel Symbol Test:')
                lines.append(f"  - Overall: {novel_latest.get('correct','-')}/{novel_latest.get('num_tests','-')} (acc={(novel_latest.get('accuracy',0)*100):.1f}%)")
                if 'a1_to_a2_accuracy' in novel_latest:
                    lines.append(f"  - A1→A2: {(novel_latest.get('a1_to_a2_accuracy',0)*100):.1f}%  |  A2→A1: {(novel_latest.get('a2_to_a1_accuracy',0)*100):.1f}%")
                if 'ges1_ma' in novel_latest or 'ges2_ma' in novel_latest:
                    lines.append(f"  - GES (MA): A1={novel_latest.get('ges1_ma','-')}, A2={novel_latest.get('ges2_ma','-')}")

            y = 0.95
            for line in lines:
                ax.text(0.02, y, line, va='top', ha='left', fontsize=10, family='monospace', transform=ax.transAxes)
                y -= 0.03
                if y < 0.05:
                    break  # avoid overflow on the first page
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Latest plot image (if any)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('off')
            if latest_plot_path and os.path.exists(latest_plot_path):
                try:
                    img = plt.imread(latest_plot_path)
                    ax.imshow(img)
                    ax.set_title(f'Latest Plot: {os.path.basename(latest_plot_path)}', fontsize=12)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading plot: {e}', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No plot available', ha='center', va='center')
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Unseen history (most recent first, capped)
            if unseen_history:
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
                ax.axis('off')
                ax.set_title('Unseen Test History (oldest first)', fontsize=14, pad=10)
                plt.close(fig)
                
                def _new_unseen_page():
                    figp = plt.figure(figsize=(8.27, 11.69))
                    axp = figp.add_axes([0.08, 0.08, 0.84, 0.84])
                    axp.axis('off')
                    axp.set_title('Unseen Test History (oldest first)', fontsize=14, pad=10)
                    return figp, axp
                
                figp, axp = _new_unseen_page()
                y = 0.95
                for idx, s in enumerate(unseen_history, start=1):
                    overall = f"{s.get('correct','-')}/{s.get('num_tests','-')} ({(s.get('accuracy',0)*100):.1f}%)"
                    a12 = f"{(s.get('a1_to_a2_accuracy',0)*100):.1f}%" if 'a1_to_a2_accuracy' in s else '-'
                    a21 = f"{(s.get('a2_to_a1_accuracy',0)*100):.1f}%" if 'a2_to_a1_accuracy' in s else '-'
                    ges = f"A1={s.get('ges1_ma','-')}, A2={s.get('ges2_ma','-')}"
                    line = f"#{idx:03d}  Overall: {overall}  |  A1→A2: {a12}  |  A2→A1: {a21}  |  GES: {ges}"
                    axp.text(0.02, y, line, va='top', ha='left', fontsize=9, family='monospace', transform=axp.transAxes)
                    y -= 0.025
                    if y < 0.05:
                        pdf.savefig(figp)
                        plt.close(figp)
                        figp, axp = _new_unseen_page()
                        y = 0.95
                # Save the last (possibly partial) page
                pdf.savefig(figp)
                plt.close(figp)

            # Page 4: Novel history (most recent first, capped)
            if novel_history:
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
                ax.axis('off')
                ax.set_title('Novel Symbol Test History (oldest first)', fontsize=14, pad=10)
                plt.close(fig)
                
                def _new_novel_page():
                    figp = plt.figure(figsize=(8.27, 11.69))
                    axp = figp.add_axes([0.08, 0.08, 0.84, 0.84])
                    axp.axis('off')
                    axp.set_title('Novel Symbol Test History (oldest first)', fontsize=14, pad=10)
                    return figp, axp
                
                figp, axp = _new_novel_page()
                y = 0.95
                for idx, s in enumerate(novel_history, start=1):
                    overall = f"{s.get('correct','-')}/{s.get('num_tests','-')} ({(s.get('accuracy',0)*100):.1f}%)"
                    a12 = f"{(s.get('a1_to_a2_accuracy',0)*100):.1f}%" if 'a1_to_a2_accuracy' in s else '-'
                    a21 = f"{(s.get('a2_to_a1_accuracy',0)*100):.1f}%" if 'a2_to_a1_accuracy' in s else '-'
                    ges = f"A1={s.get('ges1_ma','-')}, A2={s.get('ges2_ma','-')}"
                    line = f"#{idx:03d}  Overall: {overall}  |  A1→A2: {a12}  |  A2→A1: {a21}  |  GES: {ges}"
                    axp.text(0.02, y, line, va='top', ha='left', fontsize=9, family='monospace', transform=axp.transAxes)
                    y -= 0.025
                    if y < 0.05:
                        pdf.savefig(figp)
                        plt.close(figp)
                        figp, axp = _new_novel_page()
                        y = 0.95
                # Save the last (possibly partial) page
                pdf.savefig(figp)
                plt.close(figp)

        @after_this_request
        def remove_file(response):
            try:
                os.remove(tmp_path)
            except Exception as e:
                log_debug(f"Error deleting temporary report file: {e}")
            # Add a header to help client name the file
            response.headers['X-Report-Filename'] = report_filename
            return response

        return send_file(tmp_path, mimetype='application/pdf', as_attachment=True, download_name=report_filename)

    except Exception as e:
        log_debug(f"Error generating report: {e}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

# --- NEW: Snapshot APIs ---
@app.route('/api/snapshot/save', methods=['POST'])
def request_snapshot_save():
    """Request the training process to save a snapshot via control file.
    Accepts optional JSON: { "name": "my_label" }
    """
    try:
        data = request.get_json(force=True) or {}
        name = str(data.get('name', '')).strip()
        # Write control file atomically with snapshot_request
        ctrl = {}
        try:
            if os.path.exists(CONTROL_FILE):
                with open(CONTROL_FILE, 'r') as f:
                    ctrl = json.load(f) or {}
        except Exception:
            ctrl = {}
        ctrl['snapshot_request'] = {'name': name} if name else True
        tmp = CONTROL_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(ctrl, f)
        os.replace(tmp, CONTROL_FILE)
        log_debug(f"Snapshot save requested (name='{name}')")
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_debug(f"Error requesting snapshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshot/list', methods=['GET'])
def list_snapshots():
    try:
        # Determine snapshot directory from current config
        out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
        snap_dir = os.path.join(out_dir, 'snapshots')
        if not os.path.isdir(snap_dir):
            return jsonify({'snapshots': []})
        files = [f for f in os.listdir(snap_dir) if f.endswith('.pt')]
        files.sort(reverse=True)
        return jsonify({'snapshots': files})
    except Exception as e:
        log_debug(f"Error listing snapshots: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshot/upload', methods=['POST'])
def upload_snapshot():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'Invalid file'}), 400
        # Ensure .pt extension
        filename = os.path.basename(file.filename)
        if not filename.endswith('.pt'):
            return jsonify({'error': 'Only .pt files are allowed'}), 400
        out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
        snap_dir = os.path.join(out_dir, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        save_path = os.path.join(snap_dir, filename)
        file.save(save_path)
        log_debug(f"Uploaded snapshot saved to {save_path}")
        return jsonify({'status': 'ok', 'filename': filename})
    except Exception as e:
        log_debug(f"Error uploading snapshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshot/select', methods=['POST'])
def select_snapshot():
    """Select a snapshot filename to be used on the ARC page (store in a small state file)."""
    try:
        data = request.get_json(force=True) or {}
        filename = data.get('filename', '')
        if not filename or not isinstance(filename, str):
            return jsonify({'error': 'filename is required'}), 400
        out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
        snap_dir = os.path.join(out_dir, 'snapshots')
        full_path = os.path.join(snap_dir, os.path.basename(filename))
        if not (os.path.isdir(snap_dir) and os.path.exists(full_path)):
            return jsonify({'error': 'snapshot not found'}), 404
        # Persist selection in a small JSON file for future use
        selection_path = os.path.join(snap_dir, 'selected_snapshot.json')
        with open(selection_path, 'w') as f:
            json.dump({'filename': os.path.basename(filename), 'selected_at': datetime.now().isoformat()}, f)
        log_debug(f"Selected snapshot: {filename}")
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_debug(f"Error selecting snapshot: {e}")
        return jsonify({'error': str(e)}), 500

# --- NEW: Snapshot inspection API ---
@app.route('/api/snapshot/inspect', methods=['POST'])
def inspect_snapshot():
    """Extract architecture and training state from snapshot"""
    try:
        data = request.get_json(force=True) or {}
        filename = data.get('filename', '')
        if not filename or not isinstance(filename, str):
            return jsonify({'error': 'filename is required'}), 400

        out_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir')) if current_config else DEFAULT_CONFIG.get('output_dir')
        snap_dir = os.path.join(out_dir, 'snapshots')
        snap_path = os.path.join(snap_dir, os.path.basename(filename))

        if not os.path.exists(snap_path):
            return jsonify({'error': 'Snapshot not found'}), 404

        snapshot = torch.load(snap_path, map_location='cpu')

        return jsonify({
            'architecture': snapshot.get('architecture', {}),
            'trainer_state': snapshot.get('trainer_state', {}),
            'meta': snapshot.get('meta', {})
        })
    except Exception as e:
        log_debug(f"Error inspecting snapshot: {e}")
        return jsonify({'error': str(e)}), 500

# --- NEW: Scores API ---


def _parse_test_summary_from_lines(lines, summary_header):
    """Parse the most recent summary block following a given header.
    Returns dict or None.
    """
    # Search from the end for the header
    header_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if summary_header in lines[i]:
            header_idx = i
            break
    # Fallback: some logs use a generic header
    if header_idx is None:
        for i in range(len(lines) - 1, -1, -1):
            if 'Summary (bidirectional):' in lines[i]:
                header_idx = i
                break
    # Helper regexes (robust to arrow variations and spacing)
    r_pair = re.compile(r"A1.*A2:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_pair_rev = re.compile(r"A2.*A1:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_overall = re.compile(r"Overall:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_ges = re.compile(r"GES.*Agent1=([0-9.\-]+),\s*Agent2=([0-9.\-]+)")
    
    # Build a scan window
    if header_idx is not None:
        window = lines[header_idx: header_idx + 12]
    else:
        # Fallback: scan the last chunk of the file for the latest summary
        window = lines[-60:]
    
    result = {}
    for line in window:
        m = r_pair.search(line)
        if m:
            result['a1_to_a2_correct'] = int(m.group(1))
            result['a1_to_a2_total'] = int(m.group(2))
            result['a1_to_a2_accuracy'] = float(m.group(3))
            continue
        m = r_pair_rev.search(line)
        if m:
            result['a2_to_a1_correct'] = int(m.group(1))
            result['a2_to_a1_total'] = int(m.group(2))
            result['a2_to_a1_accuracy'] = float(m.group(3))
            continue
        m = r_overall.search(line)
        if m:
            result['correct'] = int(m.group(1))
            result['num_tests'] = int(m.group(2))
            result['accuracy'] = float(m.group(3))
            continue
        m = r_ges.search(line)
        if m:
            result['ges1_ma'] = float(m.group(1))
            result['ges2_ma'] = float(m.group(2))
            continue
    # Require at least overall fields
    if 'accuracy' not in result:
        return None
    return result


def _parse_all_summaries_from_lines(lines, primary_header):
    """Parse all summary blocks in a log file and return a list of dicts in file order.
    Recognizes either the primary header or the generic 'Summary (bidirectional):'.
    """
    if not lines:
        return []
    r_pair = re.compile(r"A1.*A2:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_pair_rev = re.compile(r"A2.*A1:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_overall = re.compile(r"Overall:\s*(\d+)/(\d+)\s+correct\s+\(acc=([0-9.]+)\)")
    r_ges = re.compile(r"GES.*Agent1=([0-9.\-]+),\s*Agent2=([0-9.\-]+)")
    headers = []
    for i, line in enumerate(lines):
        if (primary_header in line) or ('Summary (bidirectional):' in line):
            headers.append(i)
    summaries = []
    for idx in headers:
        window = lines[idx: idx + 12]
        result = {}
        for line in window:
            m = r_pair.search(line)
            if m:
                result['a1_to_a2_correct'] = int(m.group(1))
                result['a1_to_a2_total'] = int(m.group(2))
                result['a1_to_a2_accuracy'] = float(m.group(3))
                continue
            m = r_pair_rev.search(line)
            if m:
                result['a2_to_a1_correct'] = int(m.group(1))
                result['a2_to_a1_total'] = int(m.group(2))
                result['a2_to_a1_accuracy'] = float(m.group(3))
                continue
            m = r_overall.search(line)
            if m:
                result['correct'] = int(m.group(1))
                result['num_tests'] = int(m.group(2))
                result['accuracy'] = float(m.group(3))
                continue
            m = r_ges.search(line)
            if m:
                result['ges1_ma'] = float(m.group(1))
                result['ges2_ma'] = float(m.group(2))
                continue
        if 'accuracy' in result:
            summaries.append(result)
    return summaries


def _read_tail_lines(path_candidates, max_lines=2000):
    """Read the tail lines of the first existing file in candidates.
    Returns (lines, path) or (None, None) if not found.
    """
    for p in path_candidates:
        if p and os.path.exists(p):
            try:
                # Efficient tail using deque
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    dq = deque(f, maxlen=max_lines)
                    return list(dq), p
            except Exception as e:
                log_debug(f"Error reading {p}: {e}")
                continue
    return None, None


def _read_all_lines(path, max_bytes=5*1024*1024, tail_fallback_lines=20000):
    """Read entire file if size <= max_bytes, otherwise tail as fallback.
    Returns list of lines or None on error.
    """
    try:
        if not os.path.exists(path):
            return None
        size = os.path.getsize(path)
        if size <= max_bytes:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        # Fallback: large file, read a larger tail window
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            dq = deque(f, maxlen=tail_fallback_lines)
            return list(dq)
    except Exception as e:
        log_debug(f"Error reading all lines from {path}: {e}")
        return None


@app.route('/api/phases', methods=['GET'])
def get_phases():
    """Get all phases"""
    try:
        phases = load_phases()
        return jsonify({'phases': phases})
    except Exception as e:
        log_debug(f"Error getting phases: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/phases', methods=['POST'])
def add_phase():
    """Add a new intermediate phase"""
    try:
        data = request.get_json(force=True)
        name = data.get('name', 'New Phase').strip()
        insert_after = data.get('insert_after', None)  # ID of phase to insert after
        
        phases = load_phases()
        
        # Find the highest numeric ID to generate a new unique ID
        max_id = 0
        for p in phases:
            if isinstance(p['id'], int):
                max_id = max(max_id, p['id'])
        new_id = max_id + 1
        
        new_phase = {
            'id': new_id,
            'name': name,
            'type': 'intermediate',
            'route': f'phase_{new_id}'
        }
        
        # Insert at the right position
        if insert_after is not None:
            # Find position to insert after
            insert_idx = len(phases) - 1  # Default: before last (ARC) phase
            for i, p in enumerate(phases):
                if p['id'] == insert_after:
                    insert_idx = i + 1
                    break
            phases.insert(insert_idx, new_phase)
        else:
            # Insert before the last phase (ARC)
            phases.insert(len(phases) - 1, new_phase)
        
        if save_phases(phases):
            return jsonify({'status': 'ok', 'phase': new_phase, 'phases': phases})
        else:
            return jsonify({'error': 'Failed to save phases'}), 500
    except Exception as e:
        log_debug(f"Error adding phase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/phases/<phase_id>', methods=['DELETE'])
def delete_phase(phase_id):
    """Delete a phase (only intermediate phases can be deleted)"""
    try:
        phases = load_phases()
        
        # Find and remove the phase
        phase_to_remove = None
        for i, p in enumerate(phases):
            if str(p['id']) == str(phase_id):
                if p['type'] == 'intermediate':
                    phase_to_remove = i
                    break
                else:
                    return jsonify({'error': 'Cannot delete base or final phases'}), 400
        
        if phase_to_remove is not None:
            removed = phases.pop(phase_to_remove)
            if save_phases(phases):
                return jsonify({'status': 'ok', 'removed': removed, 'phases': phases})
            else:
                return jsonify({'error': 'Failed to save phases'}), 500
        else:
            return jsonify({'error': 'Phase not found'}), 404
    except Exception as e:
        log_debug(f"Error deleting phase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/phases/<phase_id>', methods=['PUT'])
def update_phase(phase_id):
    """Update a phase name"""
    try:
        data = request.get_json(force=True)
        new_name = data.get('name', '').strip()
        
        if not new_name:
            return jsonify({'error': 'Name is required'}), 400
        
        phases = load_phases()
        
        # Find and update the phase
        updated = False
        for p in phases:
            if str(p['id']) == str(phase_id):
                p['name'] = new_name
                updated = True
                break
        
        if updated:
            if save_phases(phases):
                return jsonify({'status': 'ok', 'phases': phases})
            else:
                return jsonify({'error': 'Failed to save phases'}), 500
        else:
            return jsonify({'error': 'Phase not found'}), 404
    except Exception as e:
        log_debug(f"Error updating phase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scores', methods=['GET'])
def get_scores():
    """Return latest parsed scores for unseen and novel tests, plus optional history."""
    # Determine candidate paths in output_dir and cwd and src/
    output_dir = current_config.get('output_dir', DEFAULT_CONFIG.get('output_dir', './outputs')) if current_config else DEFAULT_CONFIG.get('output_dir', './outputs')
    unseen_candidates = [
        os.path.join(output_dir, 'unseen_testing_log.txt') if output_dir else None,
        'unseen_testing_log.txt',
        os.path.join('src', 'unseen_testing_log.txt')
    ]
    novel_candidates = [
        os.path.join(output_dir, 'novel_symbol_unseen_testing_log.txt') if output_dir else None,
        'novel_symbol_unseen_testing_log.txt',
        os.path.join('src', 'novel_symbol_unseen_testing_log.txt')
    ]
    
    unseen_lines, unseen_path = _read_tail_lines(unseen_candidates)
    novel_lines, novel_path = _read_tail_lines(novel_candidates)
    
    unseen = None
    novel = None
    unseen_mtime = None
    novel_mtime = None
    unseen_history = []
    novel_history = []
    
    if unseen_lines is not None:
        unseen = _parse_test_summary_from_lines(unseen_lines, 'Unseen testing summary')
        try:
            unseen_mtime = os.path.getmtime(unseen_path)
        except Exception:
            unseen_mtime = None
    if novel_lines is not None:
        novel = _parse_test_summary_from_lines(novel_lines, 'Novel symbol induction summary')
        try:
            novel_mtime = os.path.getmtime(novel_path)
        except Exception:
            novel_mtime = None
    
    # For history, parse the full file (or a large tail for very large files)
    if unseen_path:
        all_unseen = _read_all_lines(unseen_path)
        if all_unseen is not None:
            unseen_history = _parse_all_summaries_from_lines(all_unseen, 'Unseen testing summary')
        elif unseen_lines is not None:
            unseen_history = _parse_all_summaries_from_lines(unseen_lines, 'Unseen testing summary')
    if novel_path:
        all_novel = _read_all_lines(novel_path)
        if all_novel is not None:
            novel_history = _parse_all_summaries_from_lines(all_novel, 'Novel symbol induction summary')
        elif novel_lines is not None:
            novel_history = _parse_all_summaries_from_lines(novel_lines, 'Novel symbol induction summary')
    
    return jsonify({
        'unseen': unseen,
        'unseen_path': unseen_path,
        'unseen_mtime': unseen_mtime,
        'unseen_history': unseen_history,
        'novel': novel,
        'novel_path': novel_path,
        'novel_mtime': novel_mtime,
        'novel_history': novel_history
    })

if __name__ == '__main__':
    # Initialize config
    if os.path.exists('training_config.json'):
        with open('training_config.json', 'r') as f:
            current_config = json.load(f)
    else:
        current_config = DEFAULT_CONFIG.copy()
    
    # Initialize phases
    load_phases()
    
    log_debug("Starting enhanced training web interface...")
    print("Starting enhanced training web interface...")
    print("Open http://localhost:5001 in your browser")
    print("Debug endpoint: http://localhost:5001/api/debug")
    app.run(debug=True, host='0.0.0.0', port=5001)