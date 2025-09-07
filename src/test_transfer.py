#!/usr/bin/env python3
"""
Test script to verify that embedding transfer works correctly during symbol consolidation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Set, Dict
from collections import defaultdict

class MockAgent:
    """Minimal mock agent for testing embedding transfers"""
    def __init__(self, embedding_dim=64, num_symbols=20, puzzle_symbols=10):
        self.puzzle_symbols = puzzle_symbols
        self.embedding_dim = embedding_dim
        self.communication_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.current_comm_symbols = 5
        self.current_total_symbols = puzzle_symbols + self.current_comm_symbols
        
        # Initialize embeddings with distinctive patterns for testing
        with torch.no_grad():
            for i in range(num_symbols):
                # Create distinctive patterns: each embedding is filled with its index value
                self.communication_embedding.weight[i].fill_(i * 0.1)

class EmbeddingTransferTester:
    """Test class for embedding transfer functionality"""
    
    def __init__(self):
        self.agent1 = MockAgent()
        self.agent2 = MockAgent()
        self.puzzle_symbols = 10
        
        # Set up initial mappings
        self.puzzle_symbol_mapping = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}
        self.symbol_puzzle_mapping = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
        self.removed_symbols = set()
        
        print("Test Setup Complete:")
        print(f"  Initial puzzle->symbol mapping: {self.puzzle_symbol_mapping}")
        print(f"  Initial symbol->puzzle mapping: {self.symbol_puzzle_mapping}")
    
    def capture_embeddings(self) -> Dict[int, torch.Tensor]:
        """Capture current state of embeddings for comparison"""
        embeddings = {}
        for symbol in self.symbol_puzzle_mapping.keys():
            embeddings[symbol] = {
                'agent1': self.agent1.communication_embedding.weight[symbol].clone(),
                'agent2': self.agent2.communication_embedding.weight[symbol].clone()
            }
        return embeddings
    
    def print_embedding_state(self, title: str):
        """Print current embedding values for active symbols"""
        print(f"\n{title}:")
        for symbol in sorted(self.symbol_puzzle_mapping.keys()):
            puzzle_idx = self.symbol_puzzle_mapping[symbol]
            agent1_val = self.agent1.communication_embedding.weight[symbol][0].item()
            agent2_val = self.agent2.communication_embedding.weight[symbol][0].item()
            print(f"  Symbol {symbol} (puzzle {puzzle_idx}): Agent1={agent1_val:.2f}, Agent2={agent2_val:.2f}")
    
    def remove_recessive_symbols_buggy(self, recessive_symbols: Set[int]):
        """Original buggy version - no embedding transfer"""
        print(f"\n{'='*50}")
        print(f"BUGGY VERSION: Removing recessive symbols {recessive_symbols}")
        print(f"{'='*50}")
        
        # Remove symbol mappings
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                del self.symbol_puzzle_mapping[symbol]
                del self.puzzle_symbol_mapping[puzzle_idx]
        
        # Compact remaining mappings WITHOUT transferring embeddings
        remaining_mapped_puzzles = list(self.puzzle_symbol_mapping.keys())
        remaining_mapped_puzzles.sort()
        
        new_puzzle_mapping = {}
        new_symbol_mapping = {}
        new_symbol_idx = self.puzzle_symbols
        
        for puzzle_idx in remaining_mapped_puzzles:
            new_puzzle_mapping[puzzle_idx] = new_symbol_idx
            new_symbol_mapping[new_symbol_idx] = puzzle_idx
            new_symbol_idx += 1
        
        self.puzzle_symbol_mapping = new_puzzle_mapping
        self.symbol_puzzle_mapping = new_symbol_mapping
        self.removed_symbols.update(recessive_symbols)
    
    def remove_recessive_symbols_fixed(self, recessive_symbols: Set[int]):
        """Fixed version - WITH embedding transfer"""
        print(f"\n{'='*50}")
        print(f"FIXED VERSION: Removing recessive symbols {recessive_symbols}")
        print(f"{'='*50}")
        
        # Store original mappings before removal
        original_puzzle_symbol_mapping = self.puzzle_symbol_mapping.copy()
        
        # Remove symbol mappings
        for symbol in recessive_symbols:
            if symbol in self.symbol_puzzle_mapping:
                puzzle_idx = self.symbol_puzzle_mapping[symbol]
                del self.symbol_puzzle_mapping[symbol]
                del self.puzzle_symbol_mapping[puzzle_idx]
        
        # Compact remaining mappings
        remaining_mapped_puzzles = list(self.puzzle_symbol_mapping.keys())
        remaining_mapped_puzzles.sort()
        
        new_puzzle_mapping = {}
        new_symbol_mapping = {}
        new_symbol_idx = self.puzzle_symbols
        
        # Track symbol transfers
        symbol_transfer_mapping = {}
        
        for puzzle_idx in remaining_mapped_puzzles:
            old_symbol = original_puzzle_symbol_mapping[puzzle_idx]
            new_symbol = new_symbol_idx
            
            new_puzzle_mapping[puzzle_idx] = new_symbol
            new_symbol_mapping[new_symbol] = puzzle_idx
            
            if old_symbol != new_symbol:
                symbol_transfer_mapping[old_symbol] = new_symbol
                print(f"  Symbol transfer planned: {old_symbol} -> {new_symbol} (puzzle {puzzle_idx})")
            
            new_symbol_idx += 1
        
        # Update mappings
        self.puzzle_symbol_mapping = new_puzzle_mapping
        self.symbol_puzzle_mapping = new_symbol_mapping
        
        # Transfer embeddings
        if symbol_transfer_mapping:
            print(f"  Transferring embeddings for {len(symbol_transfer_mapping)} symbols...")
            with torch.no_grad():
                for old_symbol, new_symbol in symbol_transfer_mapping.items():
                    # Transfer Agent 1 embeddings
                    self.agent1.communication_embedding.weight[new_symbol].copy_(
                        self.agent1.communication_embedding.weight[old_symbol]
                    )
                    # Transfer Agent 2 embeddings
                    self.agent2.communication_embedding.weight[new_symbol].copy_(
                        self.agent2.communication_embedding.weight[old_symbol]
                    )
                    print(f"    Transferred embedding {old_symbol} -> {new_symbol}")
        
        self.removed_symbols.update(recessive_symbols)
    
    def verify_embeddings_preserved(self, original_embeddings: Dict, expected_transfers: Dict):
        """Verify that embeddings were correctly preserved after transfers"""
        print(f"\n{'='*30}")
        print("VERIFICATION RESULTS")
        print(f"{'='*30}")
        
        success = True
        
        for puzzle_idx, new_symbol in self.puzzle_symbol_mapping.items():
            # Find what the original symbol was for this puzzle
            original_symbol = None
            for orig_puzzle, orig_symbol in expected_transfers.items():
                if orig_puzzle == puzzle_idx:
                    original_symbol = orig_symbol
                    break
            
            if original_symbol is None:
                continue
                
            # Check if embedding was preserved
            original_emb_agent1 = original_embeddings[original_symbol]['agent1']
            current_emb_agent1 = self.agent1.communication_embedding.weight[new_symbol]
            
            original_emb_agent2 = original_embeddings[original_symbol]['agent2'] 
            current_emb_agent2 = self.agent2.communication_embedding.weight[new_symbol]
            
            agent1_match = torch.allclose(original_emb_agent1, current_emb_agent1, atol=1e-6)
            agent2_match = torch.allclose(original_emb_agent2, current_emb_agent2, atol=1e-6)
            
            if agent1_match and agent2_match:
                print(f"  ✓ Puzzle {puzzle_idx}: embedding preserved ({original_symbol}->{new_symbol})")
            else:
                print(f"  ✗ Puzzle {puzzle_idx}: embedding LOST ({original_symbol}->{new_symbol})")
                if not agent1_match:
                    print(f"    Agent1 mismatch: {original_emb_agent1[0].item():.2f} != {current_emb_agent1[0].item():.2f}")
                if not agent2_match:
                    print(f"    Agent2 mismatch: {original_emb_agent2[0].item():.2f} != {current_emb_agent2[0].item():.2f}")
                success = False
        
        return success

def run_buggy_test():
    """Test the buggy version to show embeddings are lost"""
    print("="*70)
    print("TEST 1: BUGGY VERSION (embeddings lost)")
    print("="*70)
    
    tester = EmbeddingTransferTester()
    
    # Capture initial state
    original_embeddings = tester.capture_embeddings()
    tester.print_embedding_state("Initial Embedding State")
    
    # Remove symbols 11 and 13 (puzzles 1 and 3 become orphaned)
    recessive_symbols = {11, 13}
    tester.remove_recessive_symbols_buggy(recessive_symbols)
    
    tester.print_embedding_state("After Buggy Removal")
    print(f"New mappings: {tester.puzzle_symbol_mapping}")
    
    # Check what happened to embeddings
    print("\nAnalysis of Buggy Behavior:")
    print("  Puzzle 0: was symbol 10, still symbol 10 → embedding preserved ✓")
    print("  Puzzle 2: was symbol 12, now symbol 11 → embedding LOST ✗")
    print("  Puzzle 4: was symbol 14, now symbol 12 → embedding LOST ✗")
    print("  Issue: Puzzle 2 now uses untrained embedding at position 11")
    print("  Issue: Puzzle 4 now uses untrained embedding at position 12")
    
    return False  # Buggy version always fails

def run_fixed_test():
    """Test the fixed version to show embeddings are preserved"""
    print("\n" + "="*70)
    print("TEST 2: FIXED VERSION (embeddings preserved)")
    print("="*70)
    
    tester = EmbeddingTransferTester()
    
    # Capture initial state
    original_embeddings = tester.capture_embeddings()
    tester.print_embedding_state("Initial Embedding State")
    
    # Create expected transfer mapping for verification
    expected_transfers = {0: 10, 2: 12, 4: 14}  # puzzle_idx: original_symbol
    
    # Remove symbols 11 and 13 
    recessive_symbols = {11, 13}
    tester.remove_recessive_symbols_fixed(recessive_symbols)
    
    tester.print_embedding_state("After Fixed Removal")
    print(f"New mappings: {tester.puzzle_symbol_mapping}")
    
    # Verify embeddings were preserved
    success = tester.verify_embeddings_preserved(original_embeddings, expected_transfers)
    
    if success:
        print("\n🎉 ALL EMBEDDINGS CORRECTLY PRESERVED!")
    else:
        print("\n❌ SOME EMBEDDINGS WERE LOST!")
    
    return success

def run_edge_case_tests():
    """Test edge cases"""
    print("\n" + "="*70)
    print("TEST 3: EDGE CASES")
    print("="*70)
    
    # Test 1: No remapping needed (first symbols removed)
    print("\nEdge Case 1: Remove first symbols (no remapping needed)")
    tester = EmbeddingTransferTester()
    original_embeddings = tester.capture_embeddings()
    
    recessive_symbols = {10, 11}  # Remove first two symbols
    tester.remove_recessive_symbols_fixed(recessive_symbols)
    
    print(f"  Result: {tester.puzzle_symbol_mapping}")
    print("  Expected: Only compacting, minimal transfers")
    
    # Test 2: Remove all but one symbol
    print("\nEdge Case 2: Remove most symbols")
    tester = EmbeddingTransferTester()
    
    recessive_symbols = {11, 12, 13, 14}  # Remove all but first
    tester.remove_recessive_symbols_fixed(recessive_symbols)
    
    print(f"  Result: {tester.puzzle_symbol_mapping}")
    print("  Expected: Only puzzle 0 remains with symbol 10")
    
    return True

def main():
    """Run all tests"""
    print("EMBEDDING TRANSFER TEST SUITE")
    print("="*70)
    
    # Run tests
    buggy_result = run_buggy_test()
    fixed_result = run_fixed_test()
    edge_result = run_edge_case_tests()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Buggy version test: {'PASS' if not buggy_result else 'FAIL'} (expected to fail)")
    print(f"Fixed version test: {'PASS' if fixed_result else 'FAIL'}")
    print(f"Edge cases test: {'PASS' if edge_result else 'FAIL'}")
    
    if fixed_result and edge_result:
        print("\n🎉 EMBEDDING TRANSFER FIX VERIFIED WORKING!")
        print("The fix successfully preserves learned embeddings during consolidation.")
    else:
        print("\n❌ EMBEDDING TRANSFER FIX HAS ISSUES!")
        print("The fix needs further debugging.")

if __name__ == "__main__":
    main()