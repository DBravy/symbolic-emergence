import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union, Set

class PuzzleEmbedding(nn.Module):
    """
    This version of PuzzleEmbedding now converts a puzzle grid into a sequence
    of token embeddings (one per grid cell) rather than pooling to a single vector.
    """
    def __init__(
        self,
        embedding_dim: int = 512,
        num_symbols: int = 15,
        puzzle_symbols: int = 10,
        max_grid_size: Tuple[int, int] = (30, 30),
        attention_hidden_dim: int = 256
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        self.puzzle_symbols = puzzle_symbols
        self.max_grid_size = max_grid_size

        # Embedding layers for grid cell symbols and for positions.
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.row_embedding = nn.Linear(1, embedding_dim)
        self.col_embedding = nn.Linear(1, embedding_dim)
        nn.init.xavier_uniform_(self.symbol_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.row_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.col_embedding.weight, gain=0.1)
        
        self.embed_norm = nn.LayerNorm(embedding_dim)
        
        # Grid size embedding.
        self.size_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        nn.init.xavier_uniform_(self.size_embedding[0].weight, gain=0.1)
        
        # We no longer use attention pooling here.
        self.continuous_proj = None

    def embed_puzzle(self, puzzle_grid: torch.Tensor) -> torch.Tensor:
        """Convert a puzzle grid of shape [B, H, W] into sequence of embeddings [B, H*W, embedding_dim]"""
        batch_size, height, width = puzzle_grid.shape
        device = puzzle_grid.device
        
        # Create row and column positions as float tensors
        row_pos = torch.arange(height, dtype=torch.float32, device=device)  # [H]
        col_pos = torch.arange(width, dtype=torch.float32, device=device)   # [W]
        
        # Create grid of positions
        # Need to reshape for nn.Linear which expects [N, input_dim]
        row_ids = row_pos.view(-1, 1).repeat(1, width)  # [H, W]
        col_ids = col_pos.view(1, -1).repeat(height, 1)  # [H, W]
        
        # Reshape for nn.Linear
        row_ids_flat = row_ids.view(-1, 1)  # [H*W, 1]
        col_ids_flat = col_ids.view(-1, 1)  # [H*W, 1]
        
        # Get position embeddings through linear transformation
        row_emb = self.row_embedding(row_ids_flat)  # [H*W, embedding_dim]
        col_emb = self.col_embedding(col_ids_flat)  # [H*W, embedding_dim]
        
        # Reshape back to grid form
        row_emb = row_emb.view(height, width, self.embedding_dim)  # [H, W, D]
        col_emb = col_emb.view(height, width, self.embedding_dim)  # [H, W, D]
        
        # Apply normalization
        row_emb = self.embed_norm(row_emb)
        col_emb = self.embed_norm(col_emb)
        
        # Embed the puzzle symbols (unchanged)
        if puzzle_grid.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            symbol_emb = self.symbol_embedding(puzzle_grid)  # [B, H, W, D]
        else:
            if self.continuous_proj is None:
                self.continuous_proj = nn.Linear(1, self.embedding_dim).to(device)
            symbol_emb = self.continuous_proj(puzzle_grid.unsqueeze(-1))
        symbol_emb = self.embed_norm(symbol_emb)
        
        # Embed the grid size (unchanged)
        size_tensor = torch.tensor([[height, width]], dtype=torch.float, device=device).repeat(batch_size, 1)
        size_emb = self.size_embedding(size_tensor)  # [B, D]
        size_emb = size_emb.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1)
        
        # Expand row/column embeddings to batch dimension
        row_emb = row_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        col_emb = col_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Combine all embeddings
        combined = (symbol_emb + row_emb + col_emb + size_emb) / 4.0
        combined = self.embed_norm(combined)
        
        # Reshape to sequence form
        flattened = combined.view(batch_size, height * width, self.embedding_dim)
        return flattened