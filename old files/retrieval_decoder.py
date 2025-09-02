import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from encoder import RMSNorm
from decoder import ProgressiveDecoder, GridPositionalEncoding, RefinementLayer

class RetrievalProgressiveDecoder(ProgressiveDecoder):
    """
    Extended decoder that can operate in two modes:
    1. Retrieval mode: Select puzzle from database based on message
    2. Generation mode: Generate puzzle from scratch (original behavior)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, puzzle_symbols: int,
                 max_height: int = 30, max_width: int = 30, num_refinements: int = 3,
                 puzzle_database: Optional[List[torch.Tensor]] = None):
        super().__init__(embedding_dim, hidden_dim, puzzle_symbols, max_height, max_width, num_refinements)
        
        # Retrieval-specific components
        self.puzzle_database = puzzle_database if puzzle_database is not None else []
        self.puzzle_embeddings = None  # Will be computed when database is set
        self.retrieval_mode = False
        
        # Target embedding predictor - converts message to target puzzle embedding
        self.target_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, embedding_dim),  # Match puzzle embedding dimension
            nn.LayerNorm(embedding_dim)
        )
        
        # Confidence predictor for retrieval quality
        self.retrieval_confidence = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for similarity score
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize target predictor
        for layer in self.target_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Smaller gain for stability
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def set_puzzle_database(self, puzzle_database: List[torch.Tensor], puzzle_embeddings: torch.Tensor):
        """Set the puzzle database and pre-computed embeddings"""
        self.puzzle_database = puzzle_database
        self.puzzle_embeddings = puzzle_embeddings
        print(f"Set puzzle database with {len(puzzle_database)} puzzles")
    
    def set_retrieval_mode(self, enabled: bool):
        """Enable or disable retrieval mode"""
        self.retrieval_mode = enabled
        print(f"Retrieval mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def _create_perfect_logits(self, puzzle: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Convert a puzzle tensor to perfect logits (high values for correct symbols, low for others)
        
        Args:
            puzzle: [height, width] tensor with integer values
            batch_size: Number of batches to create
            
        Returns:
            logits: [batch_size, height, width, puzzle_symbols] with perfect predictions
        """
        height, width = puzzle.shape
        device = puzzle.device
        
        # Create logits tensor
        logits = torch.full((batch_size, height, width, self.puzzle_symbols), 
                           -10.0, device=device, dtype=torch.float32)
        
        # Set high values for correct symbols
        for h in range(height):
            for w in range(width):
                symbol = puzzle[h, w].item()
                symbol = max(0, min(symbol, self.puzzle_symbols - 1))  # Clamp to valid range
                logits[:, h, w, symbol] = 10.0
        
        return logits
    
    def _retrieval_forward(self, message: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass in retrieval mode - select puzzle from database
        
        Returns same format as parent class for compatibility
        """
        batch_size = message.size(0)
        device = message.device
        
        if len(self.puzzle_database) == 0 or self.puzzle_embeddings is None:
            print("Warning: No puzzle database available, falling back to generation mode")
            return super().forward(message, temperature)
        
        # Pool message to single vector
        message_pooled = message.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Generate target embedding - this should match the pooled puzzle embeddings
        target_embedding = self.target_predictor(message_pooled)  # [batch_size, embedding_dim]
        
        # Compute similarities with all puzzles in database
        # Normalize embeddings for stable cosine similarity
        target_embedding_norm = F.normalize(target_embedding, p=2, dim=-1)
        puzzle_embeddings_norm = F.normalize(self.puzzle_embeddings, p=2, dim=-1)
        
        # Compute cosine similarities
        similarities = torch.matmul(target_embedding_norm, puzzle_embeddings_norm.t())  # [batch_size, num_puzzles]
        
        # Select best matching puzzle for each batch item
        best_indices = similarities.argmax(dim=-1)  # [batch_size]
        best_similarities = similarities.gather(1, best_indices.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Get selected puzzles and create perfect logits
        selected_puzzles = []
        all_logits = []
        confidence_scores = []
        
        max_height = 0
        max_width = 0
        
        for i in range(batch_size):
            puzzle_idx = best_indices[i].item()
            selected_puzzle = self.puzzle_database[puzzle_idx]
            selected_puzzles.append(selected_puzzle)
            
            max_height = max(max_height, selected_puzzle.shape[0])
            max_width = max(max_width, selected_puzzle.shape[1])
        
        # Create logits for all selected puzzles (pad to common size)
        for i, selected_puzzle in enumerate(selected_puzzles):
            # Pad puzzle to max size
            height, width = selected_puzzle.shape
            padded_puzzle = torch.zeros(max_height, max_width, device=device, dtype=selected_puzzle.dtype)
            padded_puzzle[:height, :width] = selected_puzzle
            
            # Create perfect logits for this puzzle
            puzzle_logits = self._create_perfect_logits(padded_puzzle, 1)  # [1, max_height, max_width, puzzle_symbols]
            all_logits.append(puzzle_logits[0])  # Remove batch dimension
            
            # Create confidence score based on similarity
            similarity_score = best_similarities[i].unsqueeze(0)
            pooled_with_sim = torch.cat([message_pooled[i:i+1], similarity_score.unsqueeze(0)], dim=-1)
            confidence = self.retrieval_confidence(pooled_with_sim)
            confidence_scores.append(confidence)
        
        # Stack all logits
        final_logits = torch.stack(all_logits, dim=0)  # [batch_size, max_height, max_width, puzzle_symbols]
        
        # Create size logits based on selected puzzles
        height_logits = torch.zeros(batch_size, self.max_height, device=device)
        width_logits = torch.zeros(batch_size, self.max_width, device=device)
        
        for i, selected_puzzle in enumerate(selected_puzzles):
            actual_height, actual_width = selected_puzzle.shape
            # Create perfect size predictions
            if actual_height <= self.max_height:
                height_logits[i, actual_height - 1] = 10.0  # -1 because sizes are 1-based
            if actual_width <= self.max_width:
                width_logits[i, actual_width - 1] = 10.0
        
        # Create intermediate outputs (just copies of final for simplicity)
        intermediate_outputs = [final_logits.clone() for _ in range(len(self.refinement_layers))]
        
        return final_logits, intermediate_outputs, confidence_scores, (height_logits, width_logits)
    
    def forward(self, message: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass - choose between retrieval and generation based on mode
        """
        if self.retrieval_mode:
            return self._retrieval_forward(message, temperature)
        else:
            return super().forward(message, temperature)
    
    def get_retrieval_stats(self) -> dict:
        """Get statistics about the retrieval system"""
        return {
            'retrieval_mode': self.retrieval_mode,
            'database_size': len(self.puzzle_database),
            'has_embeddings': self.puzzle_embeddings is not None,
            'embedding_dim': self.puzzle_embeddings.shape[1] if self.puzzle_embeddings is not None else None
        }

def build_retrieval_decoder(embedding_dim: int, hidden_dim: int, puzzle_symbols: int, 
                           grid_size: Optional[Tuple[int, int]] = None,
                           puzzle_database: Optional[List[torch.Tensor]] = None) -> RetrievalProgressiveDecoder:
    """Build a retrieval-capable decoder"""
    max_height = grid_size[0] if grid_size else 30
    max_width = grid_size[1] if grid_size else 30
    return RetrievalProgressiveDecoder(
        embedding_dim, hidden_dim, puzzle_symbols, 
        max_height, max_width, puzzle_database=puzzle_database
    )