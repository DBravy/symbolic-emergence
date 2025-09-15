#!/usr/bin/env python3
"""
Debug script to identify issues with the web training client.
Run this to diagnose what's going wrong with the training process.
"""

import os
import sys
import json
import subprocess
import time
import signal
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("=== FILE EXISTENCE CHECK ===")
    
    files_to_check = [
        'train_selection.py',
        'src/train_selection.py', 
        'arc-agi_test_challenges.json',
        'src/arc-agi_test_challenges.json',
        'training_config.json',
        'agent_selection.py',
        'src/agent_selection.py',
        'trainer_selection.py',
        'src/trainer_selection.py'
    ]
    
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"{'✓' if exists else '✗'} {file_path} - {size} bytes")
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")

def test_config():
    """Test loading the training configuration"""
    print("\n=== CONFIGURATION TEST ===")
    
    try:
        if os.path.exists('training_config.json'):
            with open('training_config.json', 'r') as f:
                config = json.load(f)
            print("✓ Configuration loaded successfully")
            print(f"Config keys: {list(config.keys())}")
        else:
            print("✗ No training_config.json found")
    except Exception as e:
        print(f"✗ Error loading config: {e}")

def test_imports():
    """Test if we can import the required modules"""
    print("\n=== IMPORT TEST ===")
    
    modules_to_test = [
        'torch',
        'numpy', 
        'matplotlib',
        'json',
        'random'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
    
    # Test local modules
    local_modules = []
    if os.path.exists('agent_selection.py'):
        local_modules.append('agent_selection')
    elif os.path.exists('src/agent_selection.py'):
        sys.path.insert(0, 'src')
        local_modules.append('agent_selection')
    
    if os.path.exists('trainer_selection.py'):
        local_modules.append('trainer_selection')
    elif os.path.exists('src/trainer_selection.py'):
        if 'src' not in sys.path:
            sys.path.insert(0, 'src')
        local_modules.append('trainer_selection')
    
    for module in local_modules:
        try:
            __import__(module)
            print(f"✓ {module} (local)")
        except Exception as e:
            print(f"✗ {module} (local): {e}")

def test_subprocess_basic():
    """Test basic subprocess functionality"""
    print("\n=== BASIC SUBPROCESS TEST ===")
    
    try:
        # Test simple Python execution
        result = subprocess.run([sys.executable, '-c', 'print("Hello from subprocess")'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Basic subprocess works")
            print(f"Output: {result.stdout.strip()}")
        else:
            print(f"✗ Subprocess failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Subprocess error: {e}")

def test_training_script_syntax():
    """Test if the training script has syntax errors"""
    print("\n=== SYNTAX CHECK ===")
    
    script_paths = ['train_selection.py', 'src/train_selection.py']
    
    for script_path in script_paths:
        if os.path.exists(script_path):
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', script_path],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✓ {script_path} syntax OK")
                else:
                    print(f"✗ {script_path} syntax error: {result.stderr}")
                break
            except Exception as e:
                print(f"✗ Error checking {script_path}: {e}")

def test_training_script_import():
    """Test if we can import from the training script"""
    print("\n=== TRAINING SCRIPT IMPORT TEST ===")
    
    script_paths = [
        ('train_selection.py', ''),
        ('src/train_selection.py', 'src')
    ]
    
    for script_path, dir_to_add in script_paths:
        if os.path.exists(script_path):
            try:
                if dir_to_add and dir_to_add not in sys.path:
                    sys.path.insert(0, dir_to_add)
                
                # Try to import specific functions
                import importlib.util
                spec = importlib.util.spec_from_file_location("train_module", script_path)
                train_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_module)
                
                print(f"✓ Successfully imported {script_path}")
                
                # Check for key functions
                functions_to_check = ['load_arc_puzzles', 'main', 'parse_args']
                for func_name in functions_to_check:
                    if hasattr(train_module, func_name):
                        print(f"  ✓ Found function: {func_name}")
                    else:
                        print(f"  ✗ Missing function: {func_name}")
                
                break
                
            except Exception as e:
                print(f"✗ Error importing {script_path}: {e}")
                import traceback
                traceback.print_exc()

def test_training_process():
    """Test starting the training process like the web app does"""
    print("\n=== TRAINING PROCESS TEST ===")
    
    script_path = None
    if os.path.exists('src/train_selection.py'):
        script_path = 'src/train_selection.py'
    elif os.path.exists('train_selection.py'):
        script_path = 'train_selection.py'
    
    if not script_path:
        print("✗ No training script found")
        return
    
    try:
        print(f"Testing with script: {script_path}")
        
        # Start process like web app does
        cmd = [sys.executable, script_path, '--config', 'training_config.json', 
               '--status-file', 'training_status.json', '--web-mode']
        
        print(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, bufsize=1)
        
        print(f"✓ Process started with PID: {process.pid}")
        
        # Monitor for a few seconds
        start_time = time.time()
        timeout = 10  # seconds
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                print(f"✗ Process terminated early with code: {poll_result}")
                stdout, stderr = process.communicate()
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return
            
            # Check for status file updates
            if os.path.exists('training_status.json'):
                try:
                    with open('training_status.json', 'r') as f:
                        status = json.load(f)
                    print(f"Status update: {status}")
                except:
                    pass
            
            # Try to read some output
            try:
                import select
                import sys
                if hasattr(select, 'select'):  # Unix-like systems
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            print(f"STDOUT: {line.strip()}")
            except:
                pass  # Windows doesn't support select on pipes
            
            time.sleep(0.5)
        
        print("✓ Process ran for 10 seconds without crashing")
        
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
    except Exception as e:
        print(f"✗ Error testing training process: {e}")
        import traceback
        traceback.print_exc()

def test_arc_loading():
    """Test if we can load ARC puzzles"""
    print("\n=== ARC PUZZLE LOADING TEST ===")
    
    arc_paths = ['arc-agi_test_challenges.json', 'src/arc-agi_test_challenges.json']
    
    for arc_path in arc_paths:
        if os.path.exists(arc_path):
            try:
                with open(arc_path, 'r') as f:
                    data = json.load(f)
                print(f"✓ {arc_path} loaded successfully")
                print(f"  Number of puzzles: {len(data)}")
                
                # Test first puzzle structure
                first_key = list(data.keys())[0]
                first_puzzle = data[first_key]
                print(f"  First puzzle keys: {list(first_puzzle.keys())}")
                
                if 'train' in first_puzzle:
                    print(f"  Training examples: {len(first_puzzle['train'])}")
                if 'test' in first_puzzle:
                    print(f"  Test examples: {len(first_puzzle['test'])}")
                
                break
                
            except Exception as e:
                print(f"✗ Error loading {arc_path}: {e}")

def cleanup():
    """Clean up test files"""
    print("\n=== CLEANUP ===")
    
    files_to_clean = ['training_status.json']
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✓ Cleaned up {file_path}")
            except Exception as e:
                print(f"✗ Error cleaning {file_path}: {e}")

def main():
    """Run all diagnostic tests"""
    print("Training Process Diagnostic Tool")
    print("=" * 50)
    
    try:
        check_files()
        test_config()
        test_imports()
        test_subprocess_basic()
        test_training_script_syntax()
        test_training_script_import()
        test_arc_loading()
        test_training_process()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()