from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Set
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
        # Add a learnable bias to break symmetry between positions.
        self.position_bias = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1)] + self.position_bias[:x.size(1)]

class TransformerGridToMessage(nn.Module):
    """
    Converts a flattened grid (sequence of cell embeddings) into a fixed-length message.
    
    Instead of using learned queries, we pool the encoded sequence by selecting tokens 
    from fixed positions (evenly spaced) so that token 0 corresponds roughly to the top-left 
    of the grid and the last token to the bottom-right.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_symbols: int,
        max_seq_length: int,
        num_layers: int = 1,    # Fewer layers help preserve local details.
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        # Transformer encoder: note we use embedding_dim for both input and output.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=nhead, 
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=1000)
        self.dropout = nn.Dropout(dropout)
        # Output heads: one to predict symbol logits for each message token and
        # one to predict length logits (using mean-pooled message tokens).
        self.symbol_head = nn.Linear(embedding_dim, num_symbols)
        self.length_head = nn.Linear(embedding_dim, max_seq_length)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, embedding_dim] where L = grid height * grid width.
        B, L, D = x.shape
        x = self.pos_encoder(x)
        orig = x.clone()  # preserve the original positional–encoded input.
        x = self.encoder(x)  # [B, L, D]
        x = x + orig       # residual connection to help preserve local (position-specific) info.
        
        # Select fixed indices from 0 to L-1 evenly spaced, to obtain a message of length max_seq_length.
        indices = torch.linspace(0, L - 1, steps=self.max_seq_length).long().to(x.device)  # [max_seq_length]
        indices = indices.unsqueeze(0).expand(B, -1)  # [B, max_seq_length]
        # Expand indices so we can gather along the sequence dimension.
        indices_expanded = indices.unsqueeze(-1).expand(B, self.max_seq_length, D)
        message_tokens = x.gather(dim=1, index=indices_expanded)  # [B, max_seq_length, D]
        message_tokens = self.dropout(message_tokens)
        
        # Compute symbol logits for each message token.
        symbol_logits = self.symbol_head(message_tokens)  # [B, max_seq_length, num_symbols]
        # Pool the message tokens (via mean) to compute length logits.
        pooled = message_tokens.mean(dim=1)  # [B, D]
        length_logits = self.length_head(pooled)  # [B, max_seq_length]
        return length_logits, symbol_logits

# This is the build_encoder used by Agent.
def build_encoder(embedding_dim: int, hidden_dim: int, num_symbols: int, max_seq_length: int) -> nn.Module:
    # Here we ignore hidden_dim (since we use embedding_dim as the working dimension).
    return TransformerGridToMessage(embedding_dim, num_symbols, max_seq_length)

class DynamicDecoder(nn.Module):
    """Decoder network that can handle variable size outputs"""
    def __init__(self, embedding_dim: int, hidden_dim: int, puzzle_symbols: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Linear(hidden_dim, puzzle_symbols)
        
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size = x.shape[0]
        # Process through main network
        features = self.main(x)
        # Reshape and expand features to match target grid size
        features = features.view(batch_size, 1, 1, -1)
        features = features.expand(batch_size, height, width, -1)
        # Project to output symbols
        return self.output_proj(features)

def build_decoder(
    embedding_dim: int,
    hidden_dim: int,
    puzzle_symbols: int,
    grid_size: Optional[Tuple[int, int]] = None  # Made optional
) -> nn.Module:
    """
    Returns a decoder network that can handle variable size outputs.
    For backward compatibility, if grid_size is provided, returns a fixed-size decoder.
    """
    if grid_size is not None:
        # Original fixed-size decoder for backward compatibility
        return nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size[0] * grid_size[1] * puzzle_symbols)
        )
    else:
        # New dynamic decoder
        return DynamicDecoder(embedding_dim, hidden_dim, puzzle_symbols)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, puzzle_symbols: int):
        super().__init__()
        self.puzzle_symbols = puzzle_symbols
        
        # Initialize threshold with a very low value to encourage non-zeros initially
        self.change_threshold = nn.Parameter(torch.tensor(-1.0))  # Will sigmoid to ~0.27
        
        # Rest of decoder same as before
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.change_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.symbol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, puzzle_symbols)
        )
        
        self._initialize_weights()
        
        # Freeze all parameters except threshold
        for param in self.layers.parameters():
            param.requires_grad = True
        for param in self.change_head.parameters():
            param.requires_grad = True
        for param in self.symbol_head.parameters():
            param.requires_grad = True
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.symbol_head[-1]:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    m.bias.data[0] = -2.0
                    if m.bias.data.shape[0] > 1:
                        m.bias.data[1:] = 0.1
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Process through main network
        features = self.layers(x)
        
        # Reshape if needed
        if height is not None and width is not None:
            features = features.view(batch_size, 1, 1, -1)
            features = features.expand(batch_size, height, width, -1)
        
        # Get logits
        change_logits = self.change_head(features)
        symbol_logits = self.symbol_head(features)
        
        # Compute change probabilities
        change_probs = torch.sigmoid(change_logits)
        
        # Create zero baseline
        zero_baseline = torch.zeros_like(symbol_logits)
        zero_baseline[..., 0] = 1.0
        
        # Get current threshold
        effective_threshold = torch.sigmoid(self.change_threshold)
        
        # Debug info (store in state)
        self.last_debug_info = {
            'threshold_raw': self.change_threshold.item(),
            'threshold_effective': effective_threshold.item(),
            'change_probs_mean': change_probs.mean().item(),
            'change_probs_std': change_probs.std().item(),
            'symbol_logits_mean': symbol_logits.mean().item(),
            'symbol_logits_std': symbol_logits.std().item(),
            'percent_above_threshold': (change_probs > effective_threshold).float().mean().item() * 100
        }
        
        # Combine predictions
        final_logits = torch.where(
            change_probs > effective_threshold,
            symbol_logits,
            torch.log(zero_baseline + 1e-10)
        )
        
        return final_logits

    def decode_with_attention(self, x: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, Dict]:
        """Forward pass that returns attention info for visualization"""
        features = self.layers(x)
        features = features.view(-1, 1, 1, features.size(-1))
        features = features.expand(-1, height, width, -1)
        
        change_logits = self.change_head(features)
        symbol_logits = self.symbol_head(features)
        change_probs = torch.sigmoid(change_logits * 2.0)
        
        attention_info = {
            'change_probabilities': change_probs,
            'symbol_distributions': F.softmax(symbol_logits / self.base_temp, dim=-1)
        }
        
        return self.forward(x, height, width), attention_info

def build_residual_decoder(
    embedding_dim: int,
    hidden_dim: int,
    puzzle_symbols: int,
    grid_size: Optional[Tuple[int, int]] = None
) -> nn.Module:
    """Helper function to create a residual decoder"""
    return ResidualDecoder(embedding_dim, hidden_dim, puzzle_symbols)

# Optional regularization losses for the residual decoder
def compute_sparsity_loss(change_probs: torch.Tensor, target_sparsity: float = 0.1) -> torch.Tensor:
    """Encourage sparse changes from baseline"""
    actual_change_ratio = change_probs.mean()
    return F.mse_loss(actual_change_ratio, torch.tensor(target_sparsity, device=change_probs.device))

def compute_confidence_loss(symbol_logits: torch.Tensor, change_probs: torch.Tensor) -> torch.Tensor:
    """Encourage high confidence when making changes"""
    symbol_probs = F.softmax(symbol_logits, dim=-1)
    max_probs = symbol_probs.max(dim=-1)[0]
    confidence_loss = -torch.mean(max_probs * change_probs)
    return confidence_loss