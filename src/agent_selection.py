from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Dict, Set
from embeddings import PuzzleEmbedding
from decoder import build_decoder
from encoder import build_encoder

import threading
import queue

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

class QueryRefinementLayer(nn.Module):
    """Refines query representation using current message token"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.token_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.update_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, query_state, token):
        # query_state: [B, D] - current query representation
        # token: [B, D] - new token embedding
        
        token = self.token_projection(token)
        
        # Concatenate current state with new token
        combined = torch.cat([query_state, token], dim=-1)
        
        # Compute update
        update = self.update_network(combined)
        
        # Residual update
        new_state = self.norm(query_state + update)
        
        return new_state

class ProgressiveSelectionAgent(nn.Module):
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
        decoder: nn.Module = None,  # Keep for backward compatibility but won't use for selection
        fixed_size: bool = False,
        sender_scale: float = 1.0,
        decoder_output_scale: float = 0.15,
        similarity_metric: str = 'cosine'  # 'cosine', 'dot', or 'learned'
    ):
        super().__init__()
        self.agent_id = agent_id
        self.sender_scale = sender_scale
        self.similarity_metric = similarity_metric
        
        # Store maximum capacities
        self.max_num_symbols = num_symbols
        self.max_seq_length = max_seq_length
        self.puzzle_symbols = puzzle_symbols
        
        # Progressive training state - will be set by trainer
        # Default values for backward compatibility
        self.current_comm_symbols = 5  # Will be overridden by set_initial_comm_symbols()
        self.current_seq_length = 1   # Start with sequence length 1
        self.current_total_symbols = puzzle_symbols + self.current_comm_symbols
        
        # Communication and puzzle vocabularies
        self.communication_vocabulary = set(range(self.current_total_symbols))
        self.puzzle_vocabulary = set(range(puzzle_symbols))
        
        self.query_state_init = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        self.query_refinement_layers = nn.ModuleList([
            QueryRefinementLayer(embedding_dim) for _ in range(3)  # Match decoder refinements
        ])
        
        # Puzzle embedding (for converting grids to continuous representations)
        self.embedding_system = PuzzleEmbedding(
            embedding_dim=embedding_dim,
            num_symbols=num_symbols,  # Use max symbols for embedding system
            puzzle_symbols=puzzle_symbols,
            max_grid_size=max_grid_size
        )
        
        # Build encoder with maximum capacity (used for both message encoding and puzzle encoding)
        self.encoder = encoder if encoder is not None else build_encoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_symbols=num_symbols,  # Max symbols
            puzzle_symbols=puzzle_symbols,
            max_seq_length=max_seq_length  # Max sequence length
        )
        
        # Keep decoder for backward compatibility (now used for background reconstruction)
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
        
        # NEW: Components for selection task
        if similarity_metric == 'learned':
            # Learned similarity function
            self.similarity_mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Message pooling for creating single message embedding
        self.message_pooling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.max_grid_size = max_grid_size
        self.fixed_size = fixed_size

        self._apply_sender_scaling()
        
        # === NEW: Background decoder training infrastructure ===
        self._decoder_train_enabled = False
        self._decoder_queue: "queue.Queue" = queue.Queue(maxsize=1000)
        self._decoder_thread: Optional[threading.Thread] = None
        self._decoder_stop_event = threading.Event()
        # Decoder optimizer trains decoder parameters only
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
        # Loss for reconstruction (cross-entropy over puzzle symbols)
        self.reconstruction_criterion = nn.CrossEntropyLoss()
        # Logging cadence for background worker
        self._decoder_log_every = 50
    
    def _start_decoder_background_worker(self):
        if self._decoder_train_enabled and self._decoder_thread is not None:
            return
        self._decoder_train_enabled = True
        self._decoder_stop_event.clear()
        def _worker():
            # Put decoder in train mode; we only train decoder here
            self.decoder.train()
            step = 0
            while not self._decoder_stop_event.is_set():
                try:
                    batch = self._decoder_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if batch is None:
                    continue
                # batch is a list of tuples: (message, target_grid)
                messages, targets = zip(*batch)
                message_tensor = torch.stack(messages, dim=0)  # [B, seq, num_comm]
                target_tensor = torch.stack(targets, dim=0)    # [B, H, W]
                device = next(self.parameters()).device
                message_tensor = message_tensor.to(device)
                target_tensor = target_tensor.to(device)
                # Convert message to embeddings using current vocabulary
                num_comm = message_tensor.shape[-1]
                current_comm_embeddings = self.communication_embedding.weight[
                    self.puzzle_symbols:self.puzzle_symbols + num_comm
                ]
                embedded_message = torch.matmul(message_tensor, current_comm_embeddings)  # [B, seq, D]
                # Predict without forcing size; decoder must learn size from the message
                logits, _, _, (height_logits, width_logits) = self.decoder(embedded_message, temperature=1.0)
                # Compute reconstruction loss on the overlapping region if sizes differ
                B, Hp, Wp, C = logits.shape
                Ht, Wt = int(target_tensor.shape[1]), int(target_tensor.shape[2])
                Hc, Wc = min(Hp, Ht), min(Wp, Wt)
                logits_c = logits[:, :Hc, :Wc, :]
                targets_c = target_tensor[:, :Hc, :Wc]
                recon_loss = self.reconstruction_criterion(
                    logits_c.reshape(B * Hc * Wc, C),
                    targets_c.reshape(B * Hc * Wc)
                )
                # Auxiliary size prediction losses
                height_target_idx = torch.tensor([max(1, min(Ht, self.decoder.max_height)) - 1], device=device)
                width_target_idx = torch.tensor([max(1, min(Wt, self.decoder.max_width)) - 1], device=device)
                height_loss = F.cross_entropy(height_logits, height_target_idx)
                width_loss = F.cross_entropy(width_logits, width_target_idx)
                loss = recon_loss + 0.1 * (height_loss + width_loss)
                self.decoder_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.decoder_optimizer.step()
                step += 1
                # Periodic console logging
                if step % max(1, self._decoder_log_every) == 0:
                    try:
                        print(
                            f"[DecoderBG {self.agent_id}] step={step} "
                            f"recon_loss={recon_loss.item():.4f} "
                            f"size_loss={(height_loss.item()+width_loss.item()):.4f} "
                            f"pred_size={Hp}x{Wp} target={Ht}x{Wt} "
                            f"queue={self._decoder_queue.qsize()}"
                        )
                    except Exception:
                        pass
        # Use daemon thread so it doesn't block process exit
        self._decoder_thread = threading.Thread(target=_worker, daemon=True)
        self._decoder_thread.start()
    
    def enable_background_decoder_training(self):
        """
        Public API to enable background decoder training.
        Safe to call multiple times.
        """
        print(f"[DecoderBG {self.agent_id}] Enabling background decoder training")
        self._start_decoder_background_worker()
    
    def stop_background_decoder_training(self):
        self._decoder_train_enabled = False
        self._decoder_stop_event.set()
        if self._decoder_thread is not None:
            self._decoder_thread.join(timeout=2.0)
            self._decoder_thread = None
    
    def enqueue_successful_reconstruction(self, message: torch.Tensor, target_grid: torch.Tensor):
        """
        Enqueue a successful communication sample for background decoder training.
        Args:
            message: [1, seq_len, num_comm] symbol probabilities (sender→receiver message actually used)
            target_grid: [1, H, W] target puzzle grid
        """
        if not self._decoder_train_enabled:
            return
        try:
            # Standardize shapes and move to CPU for queueing
            msg = message.detach().cpu().squeeze(0)
            tgt = target_grid.detach().cpu().squeeze(0).long()
            # Small batches for better GPU utilization in worker
            self._decoder_queue.put_nowait([(msg, tgt)])
        except queue.Full:
            pass
    
    def set_initial_comm_symbols(self, initial_comm_symbols: int):
        """
        NEW: Set the initial number of communication symbols.
        This should be called by the trainer during initialization.
        
        Args:
            initial_comm_symbols: Number of communication symbols to start with
        """
        self.current_comm_symbols = min(initial_comm_symbols, self.max_num_symbols - self.puzzle_symbols)
        self.current_total_symbols = self.puzzle_symbols + self.current_comm_symbols
        
        # Update communication vocabulary
        self.communication_vocabulary = set(range(self.current_total_symbols))
        
        print(f"[{self.agent_id}] Initial communication symbols set to: {self.current_comm_symbols}")
        print(f"[{self.agent_id}] Total symbols: {self.current_total_symbols}")
    
    def is_sender(self) -> bool:
        return 'sender' in self.agent_id.lower()

    def _apply_sender_scaling(self):
        sender_components = [
            'encoder',
            'embedding_system',
            'communication_embedding',
            'message_pooling',
            'similarity_mlp'  # Include similarity components
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

    def set_vocabulary_size(self, new_comm_symbols: int):
        """
        Set the current vocabulary size (for phase-based training).
        
        Args:
            new_comm_symbols: New number of communication symbols
        """
        old_comm_symbols = self.current_comm_symbols
        
        # Update current capacities (with bounds checking)
        self.current_comm_symbols = min(new_comm_symbols, self.max_num_symbols - self.puzzle_symbols)
        self.current_total_symbols = self.puzzle_symbols + self.current_comm_symbols
        
        # Update communication vocabulary
        self.communication_vocabulary = set(range(self.current_total_symbols))
        
        print(f"[{self.agent_id}] Vocabulary updated:")
        print(f"  Communication symbols: {old_comm_symbols} → {self.current_comm_symbols}")
        print(f"  Total symbols: {self.puzzle_symbols + old_comm_symbols} → {self.current_total_symbols}")

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

    def remove_symbols(self, symbols_to_remove: Set[int]):
        """
        Remove specific symbols from the vocabulary.
        
        Args:
            symbols_to_remove: Set of symbol indices to remove
        """
        print(f"[{self.agent_id}] Removing symbols: {symbols_to_remove}")
        
        # Filter out symbols that are not in current vocabulary
        valid_symbols_to_remove = {s for s in symbols_to_remove 
                                 if self.puzzle_symbols <= s < self.current_total_symbols}
        
        if not valid_symbols_to_remove:
            print(f"[{self.agent_id}] No valid symbols to remove")
            return
        
        # Calculate new vocabulary size
        new_comm_symbols = self.current_comm_symbols - len(valid_symbols_to_remove)
        
        if new_comm_symbols < 1:
            print(f"[{self.agent_id}] Warning: Would remove all communication symbols, keeping at least 1")
            new_comm_symbols = 1
        
        # Update vocabulary size
        old_comm_symbols = self.current_comm_symbols
        self.current_comm_symbols = new_comm_symbols
        self.current_total_symbols = self.puzzle_symbols + self.current_comm_symbols
        self.communication_vocabulary = set(range(self.current_total_symbols))
        
        print(f"[{self.agent_id}] Symbols removed:")
        print(f"  Removed: {valid_symbols_to_remove}")
        print(f"  Communication symbols: {old_comm_symbols} → {self.current_comm_symbols}")
        print(f"  Total symbols: {self.puzzle_symbols + old_comm_symbols} → {self.current_total_symbols}")

    def get_position_symbol_mask(self, position: int) -> torch.Tensor:
        """
        Get a mask for which symbols are allowed at a given sequence position.
        Now allows all communication symbols at any position.
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
        """Encode puzzle to message (unchanged from original)"""
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
        
        # Apply masking for symbol selection
        masked_symbols = []
        final_symbol_logits = []
        
        for pos in range(seq_len):
            # Get position mask
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

    def encode_message_to_embedding(self, message: torch.Tensor) -> torch.Tensor:
        """
        Iteratively refine query representation with each message token.
        
        Args:
            message: [batch_size, seq_len, num_comm_symbols] - symbol probabilities
            
        Returns:
            query_embedding: [batch_size, embedding_dim] - refined query
        """
        batch_size = message.size(0)
        
        # Get communication embeddings
        current_comm_embeddings = self.communication_embedding.weight[
            self.puzzle_symbols:self.current_total_symbols
        ]
        
        # Convert message to embeddings
        message_embeddings = torch.matmul(message, current_comm_embeddings)
        # [batch, seq_len, embedding_dim]
        
        # Initialize query state
        query_state = self.query_state_init.expand(batch_size, -1)  # [B, D]
        
        # Iteratively refine with each token (NO POOLING)
        for i in range(message_embeddings.size(1)):
            token = message_embeddings[:, i, :]  # [B, D]
            query_state = self.query_refinement_layers[i % len(self.query_refinement_layers)](
                query_state, token
            )
        
        # Final query representation (from last token's refinement)
        return query_state

    def encode_puzzle_to_embedding(self, puzzle_grid: torch.Tensor) -> torch.Tensor:
        """
        NEW: Convert a puzzle directly to an embedding vector (bypassing message creation).
        This is used by the receiver to encode candidate puzzles for comparison.
        
        Args:
            puzzle_grid: [batch_size, height, width] - puzzle grid
            
        Returns:
            puzzle_embedding: [batch_size, embedding_dim] - single puzzle representation
        """
        # Get puzzle embedding using existing system
        puzzle_emb = self.embedding_system.embed_puzzle(puzzle_grid)  # [batch, seq, embedding_dim]
        
        # Pool the sequence dimension to get single embedding
        puzzle_embedding = puzzle_emb.mean(dim=1)  # Simple mean pooling
        
        return puzzle_embedding

    def compute_similarity(self, message_embedding: torch.Tensor, puzzle_embedding: torch.Tensor) -> torch.Tensor:
        """
        NEW: Compute similarity between message and puzzle embeddings.
        
        Args:
            message_embedding: [batch_size, embedding_dim]
            puzzle_embedding: [batch_size, embedding_dim] 
            
        Returns:
            similarity_score: [batch_size, 1] - similarity scores
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            message_norm = F.normalize(message_embedding, p=2, dim=-1)
            puzzle_norm = F.normalize(puzzle_embedding, p=2, dim=-1)
            similarity = torch.sum(message_norm * puzzle_norm, dim=-1, keepdim=True)
            
        elif self.similarity_metric == 'dot':
            # Dot product similarity
            similarity = torch.sum(message_embedding * puzzle_embedding, dim=-1, keepdim=True)
            
        elif self.similarity_metric == 'learned':
            # Learned similarity using MLP
            combined = torch.cat([message_embedding, puzzle_embedding], dim=-1)
            similarity = self.similarity_mlp(combined)
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity

    def select_from_candidates(
        self, 
        message: torch.Tensor, 
        candidate_puzzles: List[torch.Tensor],
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        NEW: Select the correct puzzle from a list of candidates based on the message.
        
        Args:
            message: [batch_size, seq_len, num_comm_symbols] - received message
            candidate_puzzles: List of [batch_size, height, width] - candidate puzzles
            temperature: Temperature for softmax
            
        Returns:
            selection_probs: [batch_size, num_candidates] - selection probabilities
            selection_logits: [batch_size, num_candidates] - raw similarity scores
            debug_info: Dict with intermediate representations for analysis
        """
        batch_size = message.size(0)
        num_candidates = len(candidate_puzzles)
        
        # Encode message to embedding
        message_embedding = self.encode_message_to_embedding(message)  # [batch, embed_dim]
        
        # Encode each candidate puzzle
        candidate_embeddings = []
        similarity_scores = []
        
        for candidate_puzzle in candidate_puzzles:
            # Encode candidate to embedding
            puzzle_embedding = self.encode_puzzle_to_embedding(candidate_puzzle)  # [batch, embed_dim]
            candidate_embeddings.append(puzzle_embedding)
            
            # Compute similarity
            similarity = self.compute_similarity(message_embedding, puzzle_embedding)  # [batch, 1]
            similarity_scores.append(similarity)
        
        # Stack similarities
        selection_logits = torch.cat(similarity_scores, dim=-1)  # [batch, num_candidates]
        
        # Apply temperature and softmax
        selection_probs = F.softmax(selection_logits / temperature, dim=-1)
        
        # Prepare debug info
        debug_info = {
            'message_embedding': message_embedding,
            'candidate_embeddings': torch.stack(candidate_embeddings, dim=1),  # [batch, num_candidates, embed_dim]
            'similarity_scores': selection_logits,
            'num_candidates': num_candidates
        }
        
        return selection_probs, selection_logits, debug_info

    # Keep old decode_message_to_puzzle for backward compatibility
    def decode_message_to_puzzle(   
        self,
        message: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Keep for backward compatibility but not used in selection task"""
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

    def get_vocabulary_info(self) -> Dict[str, int]:
        """Get current vocabulary information for debugging/logging"""
        return {
            'current_comm_symbols': self.current_comm_symbols,
            'current_seq_length': self.current_seq_length,
            'current_total_symbols': self.current_total_symbols,
            'max_comm_symbols': self.max_num_symbols - self.puzzle_symbols,
            'max_seq_length': self.max_seq_length,
            'puzzle_symbols': self.puzzle_symbols,
            'similarity_metric': self.similarity_metric
        }

    def print_position_symbol_mapping(self):
        """Print which symbols are used at which positions"""
        print(f"\n[{self.agent_id}] Position-Symbol Mapping (Phase-Based Selection):")
        print(f"  Similarity Metric: {self.similarity_metric}")
        print(f"  Total communication symbols: {self.current_comm_symbols}")
        print(f"  Current sequence length: {self.current_seq_length}")
        
        # For selection task, all communication symbols are available at all positions
        comm_symbol_indices = list(range(self.puzzle_symbols, self.current_total_symbols))
        
        for pos in range(self.current_seq_length):
            print(f"  Position {pos}: symbols {comm_symbol_indices} (all comm symbols available)")
        
        print(f"  Note: Phase-based training with dynamic vocabulary management")

    def freeze_all_parameters(self):
        """
        NEW: Freeze all trainable parameters in the agent.
        """
        for p in self.parameters():
            p.requires_grad = False

    def add_new_symbol_with_embedding(self, new_embedding: torch.Tensor) -> int:
        """
        NEW: Create a new communication symbol initialized at the provided embedding.
        If space remains in the `communication_embedding` table, use the next index;
        otherwise, overwrite the last available slot.
        
        Args:
            new_embedding: [D] or [1, D] tensor in the same device/dtype as embeddings
        Returns:
            symbol_index: absolute symbol index in the global vocabulary table
        """
        with torch.no_grad():
            emb = new_embedding
            if emb.dim() == 2:
                emb = emb.squeeze(0)
            emb = emb.to(self.communication_embedding.weight.device)
            emb = emb.type_as(self.communication_embedding.weight)
            # Determine next available index within max capacity
            next_idx = self.puzzle_symbols + self.current_comm_symbols
            if next_idx < self.max_num_symbols:
                # Place at next slot
                self.communication_embedding.weight[next_idx].copy_(emb)
                self.current_comm_symbols += 1
                self.current_total_symbols = self.puzzle_symbols + self.current_comm_symbols
                self.communication_vocabulary = set(range(self.current_total_symbols))
                return next_idx
            else:
                # Capacity reached, overwrite the last slot
                last_idx = self.max_num_symbols - 1
                self.communication_embedding.weight[last_idx].copy_(emb)
                # Keep counts unchanged
                return last_idx


# Create a factory function to replace the original Agent
def Agent(*args, **kwargs):
    """Factory function to create ProgressiveSelectionAgent instances"""
    return ProgressiveSelectionAgent(*args, **kwargs)