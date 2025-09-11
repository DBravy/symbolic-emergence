#!/usr/bin/env python3
"""
Symbol Tracking Diagnostic Script

Run this to debug the consolidation symbol tracking issue.
This script will trace exactly which symbols are being considered at each step.

Usage: python symbol_debug.py

Make sure your trainer object is available in the same directory or adjust the import.
"""

import torch
import random
import sys
import os
from collections import defaultdict

# Add current directory to path to import your modules
sys.path.append(os.getcwd())

def diagnose_symbol_tracking(trainer):
    """
    Comprehensive diagnosis of symbol tracking in consolidation phase
    """
    print("="*80)
    print("SYMBOL TRACKING DIAGNOSTIC REPORT")
    print("="*80)
    
    # 1. Current State Snapshot
    print("\n1. CURRENT TRAINER STATE")
    print("-" * 40)
    print(f"Active puzzles: {len(trainer.active_puzzles)}")
    print(f"Puzzle -> Symbol mapping: {trainer.puzzle_symbol_mapping}")
    print(f"Symbol -> Puzzle mapping: {trainer.symbol_puzzle_mapping}")
    print(f"Agent1 current_comm_symbols: {trainer.agent1.current_comm_symbols}")
    print(f"Agent1 current_total_symbols: {trainer.agent1.current_total_symbols}")
    print(f"Agent1 puzzle_symbols: {trainer.agent1.puzzle_symbols}")
    print(f"Removed symbols: {trainer.removed_symbols}")
    
    # 2. Symbol Ranges
    print("\n2. SYMBOL ID RANGES")
    print("-" * 40)
    puzzle_symbols_range = list(range(trainer.agent1.puzzle_symbols))
    comm_symbols_range = list(range(trainer.agent1.puzzle_symbols, trainer.agent1.current_total_symbols))
    mapped_symbols = set(trainer.symbol_puzzle_mapping.keys())
    
    print(f"Puzzle symbols range: {puzzle_symbols_range}")
    print(f"Communication symbols range: {comm_symbols_range}")
    print(f"Currently mapped symbols: {sorted(mapped_symbols)}")
    print(f"Mapped but outside comm range: {sorted(mapped_symbols - set(comm_symbols_range))}")
    print(f"Comm range but unmapped: {sorted(set(comm_symbols_range) - mapped_symbols)}")
    
    # 3. Test Consolidation Trial Generation
    print("\n3. CONSOLIDATION TRIAL SIMULATION")
    print("-" * 40)
    print("Running 20 test trials to see which symbols get generated...")
    
    trainer.agent1.eval()
    symbol_generation_counts = defaultdict(int)
    
    with torch.no_grad():
        for trial in range(20):
            # Mimic the consolidation test logic
            puzzle_idx = random.randint(0, len(trainer.active_puzzles) - 1)
            puzzle = trainer.active_puzzles[puzzle_idx]
            puzzle_tensor = torch.tensor(puzzle.test_input, dtype=torch.long, device=trainer.device).unsqueeze(0)
            
            # Generate symbol like in consolidation
            symbols, symbol_logits, _ = trainer.agent1.encode_puzzle_to_message(
                puzzle_tensor, temperature=0.1, deterministic=True
            )
            
            # Extract primary symbol using the same logic as consolidation
            if len(symbols.shape) == 3:
                relative_symbol = symbols[0, 0].argmax().item()
                primary_symbol = trainer.agent1.puzzle_symbols + relative_symbol
            else:
                primary_symbol = symbols[0, 0].item() if symbols.numel() > 0 else 0
            
            symbol_generation_counts[primary_symbol] += 1
            
            if trial < 5:  # Show details for first 5 trials
                print(f"  Trial {trial}: Puzzle {puzzle_idx} -> Symbol {primary_symbol}")
                print(f"    Is mapped?: {primary_symbol in trainer.symbol_puzzle_mapping}")
                print(f"    In comm range?: {trainer.agent1.puzzle_symbols <= primary_symbol < trainer.agent1.current_total_symbols}")
    
    print(f"\nSymbol generation summary (20 trials):")
    for symbol, count in sorted(symbol_generation_counts.items()):
        is_mapped = symbol in trainer.symbol_puzzle_mapping
        in_range = trainer.agent1.puzzle_symbols <= symbol < trainer.agent1.current_total_symbols
        print(f"  Symbol {symbol}: {count} times (mapped: {is_mapped}, in_range: {in_range})")
    
    trainer.agent1.train()
    
    # 4. Check what symbols would be analyzed
    print("\n4. CONSOLIDATION ANALYSIS SIMULATION")
    print("-" * 40)
    
    # Simulate the symbol_performance dict that would be created
    generated_symbols = set(symbol_generation_counts.keys())
    
    # Check what the "expected_symbols" would be
    expected_symbols = set(trainer.symbol_puzzle_mapping.keys())
    unused_symbols = expected_symbols - generated_symbols
    
    print(f"Symbols that would be generated in trials: {sorted(generated_symbols)}")
    print(f"Expected symbols (from mapping): {sorted(expected_symbols)}")
    print(f"Unused symbols (expected - generated): {sorted(unused_symbols)}")
    
    # 5. Trace the exact consolidation analysis logic
    print("\n5. ANALYSIS LOGIC TRACE")
    print("-" * 40)
    
    # Create a mock symbol_performance dict like the real function would
    mock_symbol_performance = {}
    
    # Add generated symbols
    for symbol in generated_symbols:
        mock_symbol_performance[symbol] = {
            'success_rate': 0.8,  # Mock value
            'usage_count': symbol_generation_counts[symbol],
            'success_count': int(symbol_generation_counts[symbol] * 0.8),
            'trial_results': [True] * symbol_generation_counts[symbol]
        }
    
    # Add unused symbols (this is where the bug might be)
    for symbol in unused_symbols:
        mock_symbol_performance[symbol] = {
            'success_rate': 0.0,
            'usage_count': 0,
            'success_count': 0,
            'trial_results': []
        }
    
    print("Mock symbol_performance dict would contain:")
    for symbol, data in sorted(mock_symbol_performance.items()):
        print(f"  Symbol {symbol}: {data['success_count']}/{data['usage_count']} correct")
    
    # 6. Check for stale references
    print("\n6. STALE REFERENCE CHECK")
    print("-" * 40)
    
    # Check if there are any old symbol references hanging around
    all_trainer_attrs = dir(trainer)
    symbol_related_attrs = [attr for attr in all_trainer_attrs if 'symbol' in attr.lower()]
    
    print("Trainer attributes containing 'symbol':")
    for attr in symbol_related_attrs:
        try:
            value = getattr(trainer, attr)
            if isinstance(value, (dict, set, list)) and value:
                print(f"  {attr}: {value}")
            elif isinstance(value, (int, str)):
                print(f"  {attr}: {value}")
        except:
            print(f"  {attr}: <could not access>")
    
    # 7. Agent state check
    print("\n7. AGENT SYMBOL STATE")
    print("-" * 40)
    print(f"Agent1 communication_vocabulary: {sorted(list(trainer.agent1.communication_vocabulary))}")
    print(f"Agent2 communication_vocabulary: {sorted(list(trainer.agent2.communication_vocabulary))}")
    
    # 8. Recommendations
    print("\n8. DIAGNOSTIC CONCLUSIONS")
    print("-" * 40)
    
    issues_found = []
    
    # Check for symbols outside current range being mapped
    invalid_mapped = mapped_symbols - set(comm_symbols_range)
    if invalid_mapped:
        issues_found.append(f"ISSUE: Mapped symbols outside comm range: {sorted(invalid_mapped)}")
    
    # Check for unused symbols being tracked
    if unused_symbols:
        issues_found.append(f"ISSUE: 'Unused' symbols would be analyzed: {sorted(unused_symbols)}")
    
    # Check for generation outside expected range
    unexpected_generated = generated_symbols - expected_symbols - set(comm_symbols_range)
    if unexpected_generated:
        issues_found.append(f"ISSUE: Generated symbols not in expected set: {sorted(unexpected_generated)}")
    
    if issues_found:
        print("PROBLEMS DETECTED:")
        for issue in issues_found:
            print(f"  ❌ {issue}")
    else:
        print("✅ No obvious issues detected in current state")
    
    print("\nRECOMMENDED FIXES:")
    print("1. In identify_recessive_symbols(), only analyze symbols from symbol_performance.keys()")
    print("2. Don't add unused_symbols to symbol_performance if they're not in current mapping")
    print("3. Add assertion: all analyzed symbols should be in current comm_symbols range")
    
    print("\n" + "="*80)
    print("END DIAGNOSTIC REPORT")
    print("="*80)


def main():
    """
    Main function - modify this to load your trainer object
    """
    print("Symbol Tracking Diagnostic Script")
    print("=" * 40)
    
    # YOU NEED TO MODIFY THIS SECTION TO LOAD YOUR TRAINER
    # Option 1: If you have a saved trainer
    # trainer = torch.load('trainer.pth')
    
    # Option 2: If you need to create a trainer (modify as needed)
    try:
        # Try to import your modules
        from trainer_selection import ProgressiveSelectionTrainer
        from agent_selection import ProgressiveSelectionAgent
        from puzzle import Puzzle
        import json
        
        print("Attempting to create a minimal trainer for diagnosis...")
        print("(You may need to modify this section based on your setup)")
        
        # Create minimal agents
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent1 = ProgressiveSelectionAgent(
            agent_id="sender",
            embedding_dim=512,
            hidden_dim=1024,
            num_symbols=100,
            puzzle_symbols=10,
            max_seq_length=2,
            sender_scale=1.0,
            similarity_metric='cosine'
        ).to(device)
        
        agent2 = ProgressiveSelectionAgent(
            agent_id="receiver", 
            embedding_dim=512,
            hidden_dim=1024,
            num_symbols=100,
            puzzle_symbols=10,
            max_seq_length=2,
            sender_scale=1.0,
            similarity_metric='cosine'
        ).to(device)
        
        # Create trainer
        trainer = ProgressiveSelectionTrainer(
            agent1=agent1,
            agent2=agent2,
            learning_rate=7e-7,
            device=device,
            num_distractors=2,
            initial_puzzle_count=3,
            initial_comm_symbols=3
        )
        
        # Try to load puzzles if available
        try:
            with open('arc-agi_test_challenges.json', 'r') as f:
                data = json.load(f)
            
            # Create some test puzzles
            test_puzzles = []
            for i, (puzzle_id, puzzle_data) in enumerate(list(data.items())[:10]):
                try:
                    example = puzzle_data['train'][0]
                    puzzle = Puzzle.from_single_example(
                        torch.tensor(example['input']),
                        torch.tensor(example['output'])
                    )
                    test_puzzles.append(puzzle)
                except:
                    continue
            
            if test_puzzles:
                trainer.available_arc_puzzles = test_puzzles
                trainer.initialize_first_puzzles()
                print(f"Loaded {len(test_puzzles)} test puzzles")
            else:
                print("No puzzles loaded - creating minimal state")
                
        except FileNotFoundError:
            print("arc-agi_test_challenges.json not found - creating minimal test state")
            # Create minimal test state for diagnosis
            trainer.active_puzzles = [None, None, None]  # Placeholder
            trainer.puzzle_symbol_mapping = {0: 10, 1: 11, 2: 12}
            trainer.symbol_puzzle_mapping = {10: 0, 11: 1, 12: 2}
            trainer.removed_symbols = {13, 14}  # Simulate some removed symbols
        
        # Run diagnosis
        diagnose_symbol_tracking(trainer)
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("\nTo use this script:")
        print("1. Make sure trainer_selection.py and agent_selection.py are in the current directory")
        print("2. Or modify the main() function to load your existing trainer object")
        print("3. Example: trainer = torch.load('your_trainer.pth')")
        
    except Exception as e:
        print(f"Error creating trainer: {e}")
        print("\nYou may need to modify the main() function to match your setup")
        print("Or load an existing trainer object instead of creating a new one")


if __name__ == "__main__":
    main()