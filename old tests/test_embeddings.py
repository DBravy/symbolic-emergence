import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from embeddings import PuzzleEmbedding

def test_embedding_dimensions():
    embedding = PuzzleEmbedding(embedding_dim=512)
    
    # Test puzzle embedding
    puzzle = torch.randint(0, 10, (2, 5, 5), dtype=torch.long)  # batch_size=2, 5x5 grid
    puzzle_emb = embedding.embed_puzzle(puzzle)
    assert puzzle_emb.shape == (2, 512)
    
    # Test message embedding
    message = torch.randint(0, 100, (2, 10), dtype=torch.long)  # batch_size=2, seq_length=10
    message_emb = embedding.embed_message(message)
    assert message_emb.shape == (2, 512)

def test_embedding_similarity():
    embedding = PuzzleEmbedding(embedding_dim=512)
    
    # Create identical puzzles
    puzzle1 = torch.randint(0, 10, (1, 3, 3), dtype=torch.long)
    puzzle2 = puzzle1.clone()
    
    emb1 = embedding.embed_puzzle(puzzle1)
    emb2 = embedding.embed_puzzle(puzzle2)
    
    similarity = embedding.embedding_similarity(emb1, emb2)
    assert similarity.item() > 0.99  # Should be very similar

def test_different_grids():
    embedding = PuzzleEmbedding(embedding_dim=512)
    
    # Create different puzzles
    puzzle1 = torch.zeros((1, 3, 3), dtype=torch.long)
    puzzle2 = torch.ones((1, 3, 3), dtype=torch.long)
    
    emb1 = embedding.embed_puzzle(puzzle1)
    emb2 = embedding.embed_puzzle(puzzle2)
    
    similarity = embedding.embedding_similarity(emb1, emb2)
    assert similarity.item() < 0.99  # Should be less similar