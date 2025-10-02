# Input Grid Initialization for Reconstruction

## Overview
Modified the reconstruction process so that the decoder **always starts with the input grid** instead of a random embedding. This means agents must learn to transform the input grid into the output grid based on the received symbol.

## What Changed

### Before
```
Input Grid → Encoder → Symbol
Symbol → Decoder (random init) → Output Grid
```

### After
```
Input Grid → Encoder → Symbol
Symbol + Input Grid → Decoder (input grid init) → Output Grid
```

## Modified Files

### 1. `src/decoder.py`
- Added `input_grid_embedding` layer to embed input grid symbols into hidden dimension
- Modified `forward()` method to accept optional `input_grid` parameter
- When `input_grid` is provided, the decoder initializes with the embedded input grid instead of random noise
- The embedded input grid is padded/cropped to match the target output size
- Positional encoding is still applied on top

**Key changes:**
```python
# New embedding layer for input grid symbols
self.input_grid_embedding = nn.Embedding(puzzle_symbols, hidden_dim)

# In forward():
if input_grid is not None:
    # Embed the input grid and use it as initialization
    embedded_input = self.input_grid_embedding(input_grid)
    # Resize to target size, then add positional encoding
    grid = ... # (see implementation)
else:
    # Original behavior: random embedding
    grid = self.grid_embedding.expand(...)
```

### 2. `src/agent.py` & `src/agent_selection.py`
- Modified `decode_message_to_puzzle()` method to accept optional `input_grid` parameter
- Passes `input_grid` to decoder when provided
- Maintains backward compatibility (works without input_grid)

**Key changes:**
```python
def decode_message_to_puzzle(
    self,
    message: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    temperature: float = 1.0,
    hard: bool = True,
    input_grid: Optional[torch.Tensor] = None  # NEW
):
    # ... 
    decoder_output = self.decoder(
        embedded_message,
        temperature=temperature,
        input_grid=input_grid  # Pass through
    )
```

### 3. `src/trainer_selection.py`
- Updated `_train_reconstruction_step()` to pass `input_tensor` to decoder
- Updated both bidirectional training directions (A1→A2 and A2→A1)
- Updated reconstruction sampling for visualization

**Key changes:**
```python
# Direction A1 -> A2
logits1, _, _, (hlog1, wlog1) = self.agent2.decoder(
    embedded_msg1, 
    temperature=1.0, 
    input_grid=input_tensor  # NEW
)

# Direction A2 -> A1  
logits2, _, _, (hlog2, wlog2) = self.agent1.decoder(
    embedded_msg2, 
    temperature=1.0, 
    input_grid=input_tensor  # NEW
)
```

### 4. `src/train_single_puzzle.py`
- Updated training loop to pass `input_tensor` to decoder
- Updated testing/evaluation to pass `test_tensor` to decoder

**Key changes:**
```python
# Training
decoded_logits, _, _, _ = agent2.decoder(
    embedded_msg, 
    temperature=1.0, 
    force_target_size=target_size, 
    input_grid=input_tensor  # NEW
)

# Testing
decoded_logits, _, _, _ = agent2.decoder(
    embedded_msg, 
    temperature=0.1, 
    force_target_size=target_size, 
    input_grid=test_tensor  # NEW
)
```

## How It Works

1. **Encoder Phase**: Agent encodes the INPUT grid into a symbol/message
   - This symbol represents "what transformation to apply"

2. **Decoder Phase**: Agent decodes the symbol to reconstruct the OUTPUT grid
   - Decoder receives both the symbol AND the input grid
   - Input grid is embedded into the same hidden dimension space
   - The embedded input grid is used as the initial state (instead of random noise)
   - If output size differs from input size, the input is padded with zeros
   - The decoder refines this initial state using the symbol information

3. **Training**: Agents learn to:
   - Encode transformation information into symbols
   - Decode symbols to modify the input grid into the output grid

## Benefits

1. **More Realistic Task**: Agents must learn to apply transformations to the input, which is closer to how ARC puzzles actually work

2. **Better Inductive Bias**: Starting from the input grid provides a strong prior - many ARC puzzles involve local modifications to the input

3. **Easier Learning**: Agents don't need to reconstruct the entire output from scratch, just the changes needed

4. **Size Handling**: When input and output sizes differ, the input is used as a starting point and extended/cropped as needed

## Backward Compatibility

The `input_grid` parameter is **optional** in all modified methods. If not provided, the decoder falls back to the original behavior (random initialization). This ensures existing code continues to work without modification.

## Usage Example

```python
# With input grid (NEW)
input_tensor = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
symbols = agent1.encode_puzzle_to_message(input_tensor)
output_logits = agent2.decoder(symbols, input_grid=input_tensor)

# Without input grid (LEGACY - still works)
output_logits = agent2.decoder(symbols)  # Uses random initialization
```

## Testing

To test the modification, you can:

1. Run single puzzle training:
   ```bash
   python src/train_single_puzzle.py --puzzle_id <puzzle_id>
   ```

2. Observe that reconstruction now starts with the input grid visible in the initial decoder state

3. Monitor reconstruction accuracy - it should improve faster since the decoder has more information

## Future Improvements

Potential enhancements to consider:

1. **Attention Masking**: Allow the symbol to indicate which parts of the input to modify
2. **Explicit Delta Encoding**: Have the decoder output only the changes to apply
3. **Multi-Step Refinement**: Use multiple symbols to iteratively refine the transformation
4. **Input-Output Alignment**: Learn explicit correspondences between input and output positions

