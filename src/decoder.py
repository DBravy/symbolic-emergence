import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from encoder import RMSNorm

class GridPositionalEncoding(nn.Module):
    """2D positional encoding for grid positions"""
    def __init__(self, d_model: int, max_height: int = 30, max_width: int = 30):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D position encoding")
        d_model_quarter = d_model // 4
        position_h = torch.arange(max_height).unsqueeze(1)
        position_w = torch.arange(max_width).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_quarter, dtype=torch.float) * (-math.log(10000.0) / d_model_quarter))
        pe_h = torch.zeros(max_height, d_model_quarter * 2)
        pe_w = torch.zeros(max_width, d_model_quarter * 2)
        pe_h[:, 0::2] = torch.sin(position_h.float() * div_term)
        pe_h[:, 1::2] = torch.cos(position_h.float() * div_term)
        pe_w[:, 0::2] = torch.sin(position_w.float() * div_term)
        pe_w[:, 1::2] = torch.cos(position_w.float() * div_term)
        pe_h = pe_h.unsqueeze(1).expand(-1, max_width, -1)
        pe_w = pe_w.unsqueeze(0).expand(max_height, -1, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, height, width, d_model]
        return x + self.pe[:x.size(1), :x.size(2)]

class RefinementLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message processing
        self.message_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output processing
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Confidence prediction
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def compute_attention(
        self, 
        query: torch.Tensor,      # [B, HW, D]
        key: torch.Tensor,        # [B, 1, D]
        value: torch.Tensor,      # [B, 1, D]
        scale: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        if scale is None:
            scale = 1.0 / math.sqrt(self.hidden_dim)
            
        # Project query, key, value
        q = self.q_proj(query)    # [B, HW, D]
        k = self.k_proj(key)      # [B, 1, D]
        v = self.v_proj(value)    # [B, 1, D]
            
        # Compute scaled dot-product attention
        attn_scores = torch.bmm(q, k.transpose(-2, -1)) * scale  # [B, HW, 1]
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, HW, 1]
        
        # Compute weighted sum
        attended = torch.bmm(attn_weights.transpose(-2, -1), q)  # [B, 1, D]
        
        return attended, attn_weights

    def forward(self, grid: torch.Tensor, message_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, D = grid.size()
        
        # Process message
        message = self.message_projection(message_token)
        
        # Prepare grid for attention
        grid_flat = grid.view(B, H*W, D)
        
        # Compute attention
        attended, attn_weights = self.compute_attention(
            query=grid_flat,
            key=message,
            value=message
        )
        
        # Process attention output
        update = self.output_proj(attended)
        update = update.view(B, 1, 1, D).expand(B, H, W, D)
        updated_grid = self.norm(grid + update)
        
        # Predict confidence based on the updated grid state
        grid_encoding = updated_grid.mean(dim=(1, 2))  # Average over spatial dimensions
        confidence = self.confidence_predictor(grid_encoding)
        
        return updated_grid, confidence

class ProgressiveDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, puzzle_symbols: int,
                 max_height: int = 30, max_width: int = 30, num_refinements: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.puzzle_symbols = puzzle_symbols
        self.max_height = max_height
        self.max_width = max_width
        
        # Project message tokens to hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Grid positional encoding
        self.pos_encoder = GridPositionalEncoding(hidden_dim, max_height, max_width)
        
        # Add size prediction network
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, max_height + max_width)  # Outputs logits for height and width
        )
        
        # Learnable initial grid embedding
        self.grid_embedding = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)
        
        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            RefinementLayer(hidden_dim) for _ in range(num_refinements)
        ])
        
        # Output projection
        self.output = nn.Linear(hidden_dim, puzzle_symbols)

    def forward(self, message: torch.Tensor, temperature: float = 1.0, force_target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = message.size(0)
        
        # Project message tokens
        message = self.input_projection(message)
        
        # Predict grid size from message
        # Pool message sequence to single vector using mean
        message_pooled = message.mean(dim=1)  # [batch_size, hidden_dim]
        size_logits = self.size_predictor(message_pooled)  # [batch_size, max_height + max_width]
        
        # Split logits into height and width
        height_logits = size_logits[:, :self.max_height]  # [batch_size, max_height]
        width_logits = size_logits[:, self.max_height:]   # [batch_size, max_width]
        
        # Get predicted dimensions using argmax, unless a target size is provided
        if force_target_size is not None:
            height = torch.tensor([force_target_size[0]], device=message.device)
            width = torch.tensor([force_target_size[1]], device=message.device)
        else:
            height = height_logits.argmax(dim=-1) + 1  # Add 1 since sizes are 1-based
            width = width_logits.argmax(dim=-1) + 1
        
        # Initialize grid with chosen size (supports batch_size==1)
        grid = self.grid_embedding.expand(batch_size, height.item(), width.item(), self.hidden_dim)
        grid = self.pos_encoder(grid)  # Position encoding will adapt to predicted size
        
        intermediate_outputs = []
        confidence_scores = []
        
        # Process each message token sequentially
        for i in range(message.size(1)):
            token = message[:, i:i+1]
            grid, confidence = self.refinement_layers[i % len(self.refinement_layers)](grid, token)
            intermediate_logits = self.output(grid)
            intermediate_outputs.append(intermediate_logits)
            confidence_scores.append(confidence)
        
        # Final output logits
        final_logits = self.output(grid) / temperature
        
        return final_logits, intermediate_outputs, confidence_scores, (height_logits, width_logits)


def build_decoder(embedding_dim: int, hidden_dim: int, puzzle_symbols: int, 
                 grid_size: Optional[Tuple[int, int]] = None) -> nn.Module:
    max_height = grid_size[0] if grid_size else 30
    max_width = grid_size[1] if grid_size else 30
    return ProgressiveDecoder(embedding_dim, hidden_dim, puzzle_symbols, max_height, max_width)