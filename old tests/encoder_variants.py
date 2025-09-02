"""
This file contains multiple encoder variants to test potential solutions to the 50/50 probability issue.
Replace your build_encoder function in encoder.py with these alternatives to test them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional

# Import the existing components from encoder.py
from encoder import RMSNorm, PositionalEncoding, TransformerEncoderLayer

# ======== VARIANT 1: Direct Classification Approach ========
class DirectClassificationEncoder(nn.Module):
    """
    Instead of using dot-product similarity, use a direct classification head.
    This avoids the normalization that erases magnitude information.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        puzzle_symbols: int,
        max_seq_length: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_comm_symbols = num_symbols - puzzle_symbols
        
        # Transformer layers (same as original)
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
        
        # Same intermediate layer as original
        self.intermediate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # CHANGE: Direct classification head instead of similarity-based approach
        self.classification_head = nn.Linear(embedding_dim, self.num_comm_symbols)
        
        # Same length prediction as original
        self.pre_length = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.ReLU()
        )
        self.length_head = nn.Linear(embedding_dim, max_seq_length)
        
        # Initialize with standard gain instead of small gain
        for module in [self.intermediate, self.classification_head, self.pre_length]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)  # Normal gain
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, comm_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        identity = x
        
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        # No scaled residual connection
        x = self.intermediate(x) + x
        
        indices = torch.linspace(0, L - 1, steps=self.max_seq_length).long()
        indices = indices.to(x.device).unsqueeze(0).expand(B, -1)
        message_tokens = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        # CHANGE: Direct classification instead of similarity
        # Note: comm_embeddings is not used in this variant
        symbol_logits = self.classification_head(message_tokens)
        
        # Length prediction (same as original)
        length_features = message_tokens.mean(dim=1)
        length_logits = self.length_head(self.pre_length(length_features))
        
        return length_logits, symbol_logits

# ======== VARIANT 2: Remove Normalization ========
class UnnormalizedSimilarityEncoder(nn.Module):
    """
    Similar to the original encoder but without normalization,
    allowing magnitude information to contribute to similarity scores.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        puzzle_symbols: int,
        max_seq_length: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_comm_symbols = num_symbols - puzzle_symbols
        
        # Same transformer layers as original
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
        
        # Enhanced intermediate with more nonlinearity
        self.intermediate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.GELU(),  # Changed from ReLU
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Stronger embedding predictor
        self.embedding_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Same length prediction as original
        self.pre_length = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.ReLU()
        )
        self.length_head = nn.Linear(embedding_dim, max_seq_length)
        
        # Normal initialization
        for module in [self.intermediate, self.embedding_predictor, self.pre_length]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, comm_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        # Apply intermediate layer
        x = self.intermediate(x) + x
        
        indices = torch.linspace(0, L - 1, steps=self.max_seq_length).long()
        indices = indices.to(x.device).unsqueeze(0).expand(B, -1)
        message_tokens = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        # CHANGE: No normalization in similarity computation
        predicted_embeddings = self.embedding_predictor(message_tokens)
        
        # Direct matrix multiplication without normalization
        symbol_logits = torch.matmul(predicted_embeddings, comm_embeddings.t())
        
        # Length prediction (same as original)
        length_features = message_tokens.mean(dim=1)
        length_logits = self.length_head(self.pre_length(length_features))
        
        return length_logits, symbol_logits

# ======== VARIANT 3: Enhanced Similarity with Temperature ========
class EnhancedSimilarityEncoder(nn.Module):
    """
    Uses a learnable temperature parameter to control the sharpness
    of the similarity distribution and amplify differences.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        puzzle_symbols: int,
        max_seq_length: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_comm_symbols = num_symbols - puzzle_symbols
        
        # Same transformer layers as original
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
        
        # Standard intermediate layer
        self.intermediate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Embedding predictor with additional components
        self.embedding_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # CHANGE: Learnable temperature scaling for similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Start small
        
        # Same length prediction as original
        self.pre_length = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.ReLU()
        )
        self.length_head = nn.Linear(embedding_dim, max_seq_length)
        
        # Normal initialization
        for module in [self.intermediate, self.embedding_predictor, self.pre_length]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, comm_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        # Process through transformer layers 
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        # Apply intermediate
        x = self.intermediate(x) + x
        
        indices = torch.linspace(0, L - 1, steps=self.max_seq_length).long()
        indices = indices.to(x.device).unsqueeze(0).expand(B, -1)
        message_tokens = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        # Get embeddings
        predicted_embeddings = self.embedding_predictor(message_tokens)
        
        # Apply normalization but with temperature scaling
        predicted_embeddings_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
        comm_embeddings_norm = F.normalize(comm_embeddings, p=2, dim=-1)
        
        # CHANGE: Use learnable temperature scaling to amplify differences
        temperature = torch.abs(self.temperature) + 0.01  # Ensure positive
        symbol_logits = torch.matmul(predicted_embeddings_norm, comm_embeddings_norm.t()) / temperature
        
        # Length prediction (same as original)
        length_features = message_tokens.mean(dim=1)
        length_logits = self.length_head(self.pre_length(length_features))
        
        return length_logits, symbol_logits

# ======== VARIANT 4: Hybrid Approach ========
class HybridEncoder(nn.Module):
    """
    Combines multiple approaches: both direct classification
    and similarity-based classification with enhanced features.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        puzzle_symbols: int,
        max_seq_length: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_comm_symbols = num_symbols - puzzle_symbols
        
        # Same transformer layers as original
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
        
        # Enhanced intermediate
        self.intermediate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Direct classification path
        self.classification_head = nn.Linear(embedding_dim, self.num_comm_symbols)
        
        # Similarity path with stronger embedding predictor
        self.embedding_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            RMSNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Learnable weight to balance between both approaches
        self.combine_weight = nn.Parameter(torch.tensor([0.5]))
        
        # Same length prediction as original
        self.pre_length = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.ReLU()
        )
        self.length_head = nn.Linear(embedding_dim, max_seq_length)
        
        # Normal initialization 
        for module in [self.intermediate, self.embedding_predictor, self.pre_length]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, comm_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        pos_encoded = self.pos_encoder(x)
        x = x + pos_encoded / math.sqrt(D)
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        # Apply intermediate
        x = self.intermediate(x) + x
        
        indices = torch.linspace(0, L - 1, steps=self.max_seq_length).long()
        indices = indices.to(x.device).unsqueeze(0).expand(B, -1)
        message_tokens = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        # Path 1: Direct classification
        direct_logits = self.classification_head(message_tokens)
        
        # Path 2: Similarity-based approach (without normalization)
        predicted_embeddings = self.embedding_predictor(message_tokens)
        similarity_logits = torch.matmul(predicted_embeddings, comm_embeddings.t())
        
        # Combine both approaches using learnable weight
        alpha = torch.sigmoid(self.combine_weight)  # Between 0 and 1
        symbol_logits = alpha * direct_logits + (1 - alpha) * similarity_logits
        
        # Length prediction (same as original)
        length_features = message_tokens.mean(dim=1)
        length_logits = self.length_head(self.pre_length(length_features))
        
        return length_logits, symbol_logits

# Replacement build_encoder functions for each variant

def build_direct_classification_encoder(embedding_dim, hidden_dim, num_symbols, puzzle_symbols, max_seq_length):
    return DirectClassificationEncoder(
        embedding_dim=embedding_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        num_layers=2,
        nhead=8,
        dropout=0.1
    )

def build_unnormalized_similarity_encoder(embedding_dim, hidden_dim, num_symbols, puzzle_symbols, max_seq_length):
    return UnnormalizedSimilarityEncoder(
        embedding_dim=embedding_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        num_layers=2,
        nhead=8,
        dropout=0.1
    )

def build_enhanced_similarity_encoder(embedding_dim, hidden_dim, num_symbols, puzzle_symbols, max_seq_length):
    return EnhancedSimilarityEncoder(
        embedding_dim=embedding_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        num_layers=2,
        nhead=8,
        dropout=0.1
    )

def build_hybrid_encoder(embedding_dim, hidden_dim, num_symbols, puzzle_symbols, max_seq_length):
    return HybridEncoder(
        embedding_dim=embedding_dim,
        num_symbols=num_symbols,
        puzzle_symbols=puzzle_symbols,
        max_seq_length=max_seq_length,
        num_layers=2,
        nhead=8,
        dropout=0.1
    )

# To use a variant, replace the build_encoder function in encoder.py with one of these