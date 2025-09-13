from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Set
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms * self.scale
        return x_normalized.view(orig_shape)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
        self.position_bias = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)] + self.position_bias[:x.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling that allows each symbol position to attend 
    to ALL grid positions with learned attention patterns.
    """
    def __init__(self, embedding_dim: int, num_heads: int = 8, max_seq_length: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable query vectors for each symbol position
        # Each symbol position learns what to "ask for" from the grid positions
        self.symbol_queries = nn.Parameter(torch.randn(max_seq_length, embedding_dim))
        
        # Layer norm for stability
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize queries with small values
        nn.init.normal_(self.symbol_queries, mean=0.0, std=0.02)
    
    def forward(self, grid_embeddings: torch.Tensor, symbol_position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool grid embeddings for a specific symbol position using multi-head attention.
        
        Args:
            grid_embeddings: [B, L, D] - embeddings for all grid positions
            symbol_position: int - which symbol position we're computing (0, 1, 2, ...)
            
        Returns:
            pooled_embedding: [B, D] - single pooled embedding for this symbol
            attention_weights: [B, num_heads, 1, L] - attention patterns (for visualization)
        """
        B, L, D = grid_embeddings.shape
        
        # Get the learnable query for this symbol position
        query = self.symbol_queries[symbol_position:symbol_position+1]  # [1, D]
        query = query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        query = self.query_norm(query)
        
        # Multi-head attention: query attends to all grid positions
        # query: [B, 1, D] - "what does this symbol want to know?"
        # grid_embeddings: [B, L, D] - keys and values (all grid positions)
        pooled, attention_weights = self.attention(
            query=query,                    # [B, 1, D]
            key=grid_embeddings,           # [B, L, D] 
            value=grid_embeddings,         # [B, L, D]
            need_weights=True,
            average_attn_weights=False     # Return per-head weights
        )
        
        # pooled: [B, 1, D], attention_weights: [B, num_heads, 1, L]
        pooled = pooled.squeeze(1)  # [B, D]
        pooled = self.output_norm(pooled)
        
        return pooled, attention_weights

class ProgressiveSimilarityEncoder(nn.Module):
    """
    Progressive encoder with multi-head attention pooling.
    Each symbol position attends to ALL grid positions with learned patterns.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        puzzle_symbols: int,
        max_seq_length: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        pooling_heads: int = 8
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.puzzle_symbols = puzzle_symbols
        self.max_num_symbols = num_symbols
        self.max_comm_symbols = num_symbols - puzzle_symbols
        
        # Transformer encoder layers
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.encoder = nn.ModuleList(encoder_layers)
        self.encoder_norm = RMSNorm(embedding_dim)
        
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=1000)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced intermediate layer
        self.intermediate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # NEW: Multi-head attention pooling instead of position sampling
        self.attention_pooling = MultiHeadAttentionPooling(
            embedding_dim=embedding_dim,
            num_heads=pooling_heads,
            max_seq_length=max_seq_length
        )
        
        # Position-specific embedding predictors
        # Each position gets its own predictor to encourage specialization
        self.position_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                RMSNorm(embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim * 2, embedding_dim)
            ) for _ in range(max_seq_length)
        ])
        
        # Position-aware learnable temperature parameters
        # Each position can have its own temperature for fine-tuning
        self.position_temperatures = nn.Parameter(torch.ones(max_seq_length) * 0.5)
        
        # Length prediction head (now uses attention pooling too)
        self.length_pooling = MultiHeadAttentionPooling(
            embedding_dim=embedding_dim,
            num_heads=4,  # Fewer heads for length prediction
            max_seq_length=1  # Only need one query for length
        )
        self.pre_length = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.ReLU()
        )
        self.length_head = nn.Linear(embedding_dim, max_seq_length)
        
        # Initialize with normal gain
        for module in [self.intermediate, self.pre_length]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize position predictors
        for predictor in self.position_predictors:
            for layer in predictor.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def get_position_symbol_embeddings(self, comm_embeddings: torch.Tensor, position: int) -> torch.Tensor:
        """
        Get the communication embeddings that are allowed for a specific position.
        Returns all communication embeddings for any position.
        
        Args:
            comm_embeddings: All communication embeddings [num_comm_symbols, embedding_dim]
            position: The sequence position (0-indexed) - unused but kept for compatibility
            
        Returns:
            All communication embeddings [num_comm_symbols, embedding_dim]
        """
        return comm_embeddings

    def forward(self, x: torch.Tensor, comm_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        # Apply positional encoding
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        # Apply intermediate layer with residual connection
        x = self.intermediate(x) + x
        
        # NEW: Use multi-head attention pooling instead of position sampling
        # Each symbol position attends to ALL grid positions with different patterns
        current_seq_length = self.max_seq_length
        symbol_logits_list = []
        pooled_embeddings = []
        
        for pos in range(current_seq_length):
            # Pool ALL grid positions for this symbol using learned attention
            pooled_embedding, attention_weights = self.attention_pooling(x, pos)  # [B, D]
            pooled_embeddings.append(pooled_embedding)
            
            # Use position-specific predictor on the pooled embedding
            predicted_embedding = self.position_predictors[pos](pooled_embedding)  # [B, D]
            
            # Get allowed embeddings for this position (all embeddings)
            pos_embeddings = self.get_position_symbol_embeddings(comm_embeddings, pos)  # [num_comm_symbols, D]
            
            if pos_embeddings.size(0) > 0:
                # Normalize embeddings
                predicted_embedding_norm = F.normalize(predicted_embedding, p=2, dim=-1)  # [B, D]
                pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=-1)  # [num_comm_symbols, D]
                
                # Calculate similarity with position-specific temperature
                temperature = torch.abs(self.position_temperatures[pos]) + 0.01
                pos_logits = torch.matmul(predicted_embedding_norm, pos_embeddings_norm.t()) / temperature  # [B, num_comm_symbols]
                
                full_logits = pos_logits
            else:
                # No valid symbols for this position
                full_logits = torch.full((B, comm_embeddings.size(0)), -1e10, 
                                    device=x.device, dtype=x.dtype)
            
            symbol_logits_list.append(full_logits)
        
        # Stack position logits
        symbol_logits = torch.stack(symbol_logits_list, dim=1)  # [B, seq_len, num_comm_symbols]
        
        # Length prediction using attention pooling
        length_pooled, _ = self.length_pooling(x, 0)  # Pool for length prediction
        length_logits = self.length_head(self.pre_length(length_pooled))
        
        return length_logits, symbol_logits
    
    def get_attention_patterns(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Get attention patterns for visualization/debugging.
        Returns attention weights for each symbol position.
        """
        B, L, D = x.shape
        
        # Process input the same way as forward()
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        x = self.intermediate(x) + x
        
        # Get attention patterns for each symbol position
        attention_patterns = {}
        for pos in range(self.max_seq_length):
            _, attention_weights = self.attention_pooling(x, pos)
            # attention_weights: [B, num_heads, 1, L]
            attention_patterns[pos] = attention_weights.squeeze(2)  # [B, num_heads, L]
        
        return attention_patterns

    def predict_symbol_embedding(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """
        NEW: Predict a symbol embedding for a given sequence position without computing logits.
        This mirrors the preprocessing in forward() and returns the position-specific predicted embedding.
        
        Args:
            x: [B, L, D] puzzle embedding sequence
            position: position index to predict embedding for (default 0)
        Returns:
            predicted_embedding: [B, D]
        """
        B, L, D = x.shape
        pos = max(0, min(position, self.max_seq_length - 1))
        
        # Same preprocessing as in forward
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        x = self.intermediate(x) + x
        
        # Pool and predict embedding for the given position
        pooled_embedding, _ = self.attention_pooling(x, pos)
        predicted_embedding = self.position_predictors[pos](pooled_embedding)
        return predicted_embedding

def build_encoder(embedding_dim: int, hidden_dim: int, num_symbols: int, puzzle_symbols: int, max_seq_length: int):
    """Build a progressive encoder with multi-head attention pooling."""
    return ProgressiveSimilarityEncoder(
        embedding_dim=embedding_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        num_layers=2,
        nhead=8,
        dropout=0.1,
        pooling_heads=8  # Number of attention heads for pooling
    )