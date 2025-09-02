from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Dict, Set
from embeddings import PuzzleEmbedding
from decoder import build_decoder
from encoder import build_encoder

class GradientScaleHook:
    def __init__(self, scale, power=1.00):
        self.scale = scale
        self.power = power

    def __call__(self, grad):
        if grad is None:
            return None
        signs = grad.sign()
        scaled = (torch.abs(grad) ** self.power) * self.scale * signs
        return scaled

class GradientAttenuationHook:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, grad):
        return grad * self.scale if grad is not None else None

class ProgressiveAgent(nn.Module):
    def __init__(
        self, 
        agent_id: str,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        num_symbols: int = 20,  # Maximum possible symbols
        puzzle_symbols: int = 10,
        max_seq_length: int = 10,  # Maximum possible sequence length
        max_grid_size: Tuple[int, int] = (30, 30),
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        fixed_size: bool = False,
        sender_scale: float = 1.0,
        decoder_output_scale: float = 0.15
    ):
        super().__init__()
        self.agent_id = agent_id
        self.sender_scale = sender_scale
        
        # Store maximum capacities
        self.max_num_symbols = num_symbols
        self.max_seq_length = max_seq_length
        self.puzzle_symbols = puzzle_symbols
        
        # Progressive training state - start small
        self.current_comm_symbols = 2  # Start with 2 communication symbols
        self.current_seq_length = 1   # Start with sequence length 1
        self.current_total_symbols = puzzle_symbols + self.current_comm_symbols
        
        # Communication and puzzle vocabularies
        self.communication_vocabulary = set(range(self.current_total_symbols))
        self.puzzle_vocabulary = set(range(puzzle_symbols))
        
        # Puzzle embedding (for converting grids to continuous representations)
        self.embedding_system = PuzzleEmbedding(
            embedding_dim=embedding_dim,
            num_symbols=num_symbols,  # Use max symbols for embedding system
            puzzle_symbols=puzzle_symbols,
            max_grid_size=max_grid_size
        )
        
        # Build encoder and decoder with maximum capacity
        self.encoder = encoder if encoder is not None else build_encoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_symbols=num_symbols,  # Max symbols
            puzzle_symbols=puzzle_symbols,
            max_seq_length=max_seq_length  # Max sequence length
        )
        self.decoder = decoder if decoder is not None else build_decoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            puzzle_symbols=puzzle_symbols,
            grid_size=max_grid_size if fixed_size else None
        )
        
        if not self.is_sender():
            self._apply_decoder_attenuation(decoder_output_scale)
            
        # Communication embedding with full capacity
        self.communication_embedding = nn.Embedding(num_symbols, embedding_dim)
        
        self.max_grid_size = max_grid_size
        self.fixed_size = fixed_size

        self._apply_sender_scaling()
    
    def is_sender(self) -> bool:
        return 'sender' in self.agent_id.lower()

    def _apply_sender_scaling(self):
        sender_components = [
            'encoder',
            'embedding_system',
            'communication_embedding'
        ]
        
        num_scaled = 0
        for name, param in self.named_parameters():
            should_scale = any(component in name for component in sender_components)
            if should_scale and param.requires_grad:
                param.register_hook(GradientScaleHook(self.sender_scale))
                num_scaled += 1

    def _apply_decoder_attenuation(self, scale):
        if hasattr(self.decoder, 'output'):
            if hasattr(self.decoder.output, 'weight') and self.decoder.output.weight.requires_grad:
                self.decoder.output.weight.register_hook(GradientAttenuationHook(scale))
            if hasattr(self.decoder.output, 'bias') and self.decoder.output.bias.requires_grad:
                self.decoder.output.bias.register_hook(GradientAttenuationHook(scale))

    def expand_vocabulary(self, additional_symbols: int = 2, additional_length: int = 1):
        """
        Expand the agent's communication vocabulary and sequence length.
        
        Args:
            additional_symbols: Number of new communication symbols to add
            additional_length: Amount to increase sequence length by
        """
        old_comm_symbols = self.current_comm_symbols
        old_seq_length = self.current_seq_length
        
        # Update current capacities (with bounds checking)
        self.current_comm_symbols = min(
            self.current_comm_symbols + additional_symbols,
            self.max_num_symbols - self.puzzle_symbols
        )
        self.current_seq_length = min(
            self.current_seq_length + additional_length,
            self.max_seq_length
        )
        self.current_total_symbols = self.puzzle_symbols + self.current_comm_symbols
        
        # Update communication vocabulary
        self.communication_vocabulary = set(range(self.current_total_symbols))
        
        print(f"[{self.agent_id}] Vocabulary expanded:")
        print(f"  Communication symbols: {old_comm_symbols} → {self.current_comm_symbols}")
        print(f"  Sequence length: {old_seq_length} → {self.current_seq_length}")
        print(f"  Total symbols: {self.puzzle_symbols + old_comm_symbols} → {self.current_total_symbols}")

    # def get_position_symbol_mask(self, position: int) -> torch.Tensor:
    #     """
    #     Get a mask for which symbols are allowed at a given sequence position.
    #     Position 0 uses symbols [puzzle_symbols, puzzle_symbols+2)
    #     Position 1 uses symbols [puzzle_symbols+2, puzzle_symbols+4)
    #     etc.
        
    #     Args:
    #         position: The sequence position (0-indexed)
            
    #     Returns:
    #         Boolean tensor indicating which symbols are allowed
    #     """
    #     device = next(self.parameters()).device
        
    #     # Calculate the symbol range for this position
    #     start_symbol = self.puzzle_symbols + position * 2
    #     end_symbol = start_symbol + 2
        
    #     # Ensure we don't exceed current vocabulary
    #     end_symbol = min(end_symbol, self.current_total_symbols)
        
    #     # Create mask for all possible symbols
    #     mask = torch.zeros(self.max_num_symbols, dtype=torch.bool, device=device)
        
    #     # Only allow symbols in the range for this position
    #     if start_symbol < self.current_total_symbols:
    #         mask[start_symbol:end_symbol] = True
            
    #     return mask

    def get_position_symbol_mask(self, position: int) -> torch.Tensor:
        """
        Get a mask for which symbols are allowed at a given sequence position.
        MODIFIED: Now allows all communication symbols at any position.
        
        Args:
            position: The sequence position (0-indexed) - now unused but kept for compatibility
            
        Returns:
            Boolean tensor indicating which symbols are allowed (all communication symbols)
        """
        device = next(self.parameters()).device
        
        # Create mask for all possible symbols
        mask = torch.zeros(self.max_num_symbols, dtype=torch.bool, device=device)
        
        # Allow all communication symbols at any position
        mask[self.puzzle_symbols:self.current_total_symbols] = True
                
        return mask

    def get_id(self) -> str:
        return self.agent_id
    
    def has_symbol(self, symbol: int) -> bool:
        return symbol in self.communication_vocabulary
    
    def has_puzzle_symbol(self, symbol: int) -> bool:
        return symbol in self.puzzle_vocabulary

    @staticmethod
    def compute_stable_probabilities(logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
        device = logits.device
        safe_temp = max(temperature, 1e-6)
        scaled_logits = logits / safe_temp
        masked_logits = torch.where(
            mask, 
            torch.clamp(scaled_logits, min=-100.0, max=100.0),
            torch.tensor(-1e5, device=device, dtype=scaled_logits.dtype)
        )
        max_logits = torch.max(
            torch.where(mask, masked_logits, torch.tensor(-1e5, device=device, dtype=masked_logits.dtype)),
            dim=-1, keepdim=True
        )[0]
        exp_logits = torch.exp(masked_logits - max_logits)
        exp_logits = torch.where(mask, exp_logits, torch.tensor(0.0, device=device, dtype=exp_logits.dtype))
        sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        probs = exp_logits / sum_exp
        probs = torch.where(mask, probs, torch.tensor(0.0, device=device, dtype=probs.dtype))
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        probs_sum = torch.clamp(probs_sum, min=1e-10)
        probs = probs / probs_sum
        return probs
    
    def gumbel_softmax(self, logits: torch.Tensor, temperature: float = 1.0, hard: bool = False, deterministic: bool = True) -> torch.Tensor:
        if deterministic:
            # Bypass Gumbel noise entirely - just use the original probabilities
            y_soft = F.softmax(logits / temperature, dim=-1)
            if hard:
                index = y_soft.max(dim=-1, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
            return ret
        
        # Original Gumbel-Softmax behavior
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret

    def encode_puzzle_to_message(self, puzzle_grid, temperature=1.0, initial_phase=False, deterministic=True):
        # Get puzzle embedding
        puzzle_emb = self.embedding_system.embed_puzzle(puzzle_grid)
        
        # Get comm embeddings for current vocabulary only
        current_comm_embeddings = self.communication_embedding.weight[
            self.puzzle_symbols:self.current_total_symbols
        ]
        
        # Get encoder outputs with current comm_embeddings
        length_logits, symbol_logits = self.encoder(puzzle_emb, current_comm_embeddings)
        
        # Truncate to current sequence length
        symbol_logits = symbol_logits[:, :self.current_seq_length, :]
        
        batch_size, seq_len, num_comm = symbol_logits.shape
        
        # Apply masking for symbol selection (now all symbols allowed at any position)
        masked_symbols = []
        final_symbol_logits = []
        
        for pos in range(seq_len):
            # Get position mask (now allows all communication symbols)
            position_mask = self.get_position_symbol_mask(pos)
            
            # Extract relevant part of mask for current communication symbols
            comm_mask = position_mask[self.puzzle_symbols:self.current_total_symbols]
            
            # Get logits for this position
            pos_logits = symbol_logits[:, pos, :]  # [batch, num_comm]
            
            # Apply mask by setting disallowed symbols to very negative values
            masked_logits = pos_logits.clone()
            masked_logits[:, ~comm_mask] = -1e10
            
            # Apply Gumbel-Softmax to masked logits with deterministic option
            pos_symbols = self.gumbel_softmax(masked_logits, temperature, hard=True, deterministic=deterministic)
            
            masked_symbols.append(pos_symbols)
            final_symbol_logits.append(masked_logits)
        
        # Stack results
        symbols = torch.stack(masked_symbols, dim=1)  # [batch, seq, num_comm]
        symbol_logits = torch.stack(final_symbol_logits, dim=1)  # [batch, seq, num_comm]
        
        return symbols, symbol_logits, {
            'total_length': symbols.size(1),
            'nonzero_symbols': (symbols.max(dim=-1)[0] > 0.5).sum().item(),
            'current_vocab_size': self.current_comm_symbols,
            'current_seq_length': self.current_seq_length
        }

    def decode_message_to_puzzle(   
        self,
        message: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if target_size is None:
            target_size = self.max_grid_size
                
        # Convert message to embeddings using current vocabulary
        current_comm_embeddings = self.communication_embedding.weight[
            self.puzzle_symbols:self.current_total_symbols
        ]
        
        # Ensure message matches current sequence length
        batch_size = message.size(0)
        if message.size(1) > self.current_seq_length:
            message = message[:, :self.current_seq_length, :]
        elif message.size(1) < self.current_seq_length:
            # Pad with zeros if needed
            padding = torch.zeros(
                batch_size, 
                self.current_seq_length - message.size(1), 
                message.size(2),
                device=message.device
            )
            message = torch.cat([message, padding], dim=1)
        
        flat_message = message.reshape(-1, message.size(-1))
        embedded_message = torch.matmul(flat_message, current_comm_embeddings)
        embedded_message = embedded_message.reshape(message.size(0), message.size(1), -1)
        
        # Get decoder outputs
        decoder_output = self.decoder(
            embedded_message,
            temperature=temperature
        )
        
        grid_logits, intermediate_logits, confidence_scores, size_logits = decoder_output
        
        # Get predicted height and width from size logits
        pred_height = size_logits[0].argmax(dim=-1) + 1
        pred_width = size_logits[1].argmax(dim=-1) + 1
        
        # Convert final output to discrete outputs using gumbel softmax
        grid = self.gumbel_softmax(
            grid_logits.reshape(-1, grid_logits.size(-1)), 
            temperature=temperature,
            hard=True
        )
        grid = grid.reshape(message.size(0), grid_logits.size(1), grid_logits.size(2), -1)
        
        # Resize grid based on predicted dimensions
        batch_size = grid.size(0)
        pred_height = torch.min(pred_height, torch.tensor(grid.size(1), device=grid.device))
        pred_width = torch.min(pred_width, torch.tensor(grid.size(2), device=grid.device))
        
        resized_grid = grid.clone()
        resized_grid_logits = grid_logits.clone()
        
        for b in range(batch_size):
            h = pred_height[b].item()
            w = pred_width[b].item()
            resized_grid[b, h:, :, :] = 0
            resized_grid[b, :, w:, :] = 0
            resized_grid_logits[b, h:, :, :] = 0
            resized_grid_logits[b, :, w:, :] = 0
        
        if len(decoder_output) >= 2:
            intermediate_logits = decoder_output[1]
            confidence_scores = decoder_output[2] if len(decoder_output) > 2 else [torch.ones(1, device=grid_logits.device)] * len(intermediate_logits)
        
        return resized_grid, resized_grid_logits, intermediate_logits, confidence_scores, size_logits

    def process_arc_example(
        self,
        input_grid: torch.Tensor,
        output_grid: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # Get message from encoder
        message, symbol_logits, _ = self.encode_puzzle_to_message(input_grid, temperature)
        
        # Get target size
        target_size = (input_grid.size(1), input_grid.size(2))
        
        # Get decoder outputs including intermediates and confidences
        reconstructed, intermediate_logits, confidence_scores = self.decode_message_to_puzzle(message, target_size)
        
        # Stack confidences into a tensor
        confidences = torch.cat(confidence_scores, dim=1)
        
        # Compute reconstruction losses
        step_weights = torch.linspace(0.5, 1.0, len(intermediate_logits), device=input_grid.device)
        recon_losses = []
        
        for step_output in intermediate_logits:
            step_loss = F.cross_entropy(
                step_output.reshape(-1, self.puzzle_symbols),
                input_grid.reshape(-1)
            )
            recon_losses.append(step_loss)
        
        # Final reconstruction loss
        final_loss = F.cross_entropy(
            reconstructed.reshape(-1, self.puzzle_symbols),
            input_grid.reshape(-1)
        )
        
        total_loss = final_loss
        
        return reconstructed, reconstructed, total_loss
    
    def evaluate_communication(
        self,
        original_puzzle: torch.Tensor,
        reconstructed_puzzle: torch.Tensor
    ) -> torch.Tensor:
        orig_emb = self.embedding_system.embed_puzzle(original_puzzle)
        recon_emb = self.embedding_system.embed_puzzle(reconstructed_puzzle)
        return self.embedding_system.embedding_similarity(orig_emb, recon_emb)

    def get_vocabulary_info(self) -> Dict[str, int]:
        """Get current vocabulary information for debugging/logging"""
        return {
            'current_comm_symbols': self.current_comm_symbols,
            'current_seq_length': self.current_seq_length,
            'current_total_symbols': self.current_total_symbols,
            'max_comm_symbols': self.max_num_symbols - self.puzzle_symbols,
            'max_seq_length': self.max_seq_length,
            'puzzle_symbols': self.puzzle_symbols
        }

    def print_position_symbol_mapping(self):
        """Print which symbols are used at which positions"""
        print(f"\n[{self.agent_id}] Position-Symbol Mapping:")
        for pos in range(self.current_seq_length):
            start_symbol = self.puzzle_symbols + pos * 2
            end_symbol = min(start_symbol + 2, self.current_total_symbols)
            available_symbols = list(range(start_symbol, end_symbol))
            print(f"  Position {pos}: symbols {available_symbols}")


# Create a factory function to replace the original Agent
def Agent(*args, **kwargs):
    """Factory function to create ProgressiveAgent instances"""
    return ProgressiveAgent(*args, **kwargs)