# Emergent Communication and Symbol Grounding in AI Agents

A research framework for studying how artificial agents develop symbolic communication systems from scratch through visual reasoning tasks. This project investigates the relationship between language, pattern recognition, and intelligence using the ARC (Abstraction and Reasoning Corpus) dataset.

## Overview

This research explores how two neural agents (sender and receiver) can learn to communicate about complex visual puzzles without pre-existing shared vocabulary. The agents must develop their own symbolic communication system through a progressive training protocol that mirrors aspects of natural language acquisition.

## Architecture

The system employs a sophisticated multi-agent architecture with several key innovations:

- **Progressive Selection Training**: Agents learn through sender→receiver selection tasks with dynamic difficulty scaling
- **Multi-Phase Learning Protocol**: Pretraining, joint training, consolidation, remedial training, and vocabulary expansion
- **Dynamic Vocabulary Management**: Automatic pruning of ineffective symbols and addition of new communication tokens
- **Transformer-Based Encoder**: Multi-head attention mechanisms for flexible visual pattern encoding
- **Novel Symbol Induction**: Testing framework for communication about completely unseen puzzles

## Key Features

### Progressive Training Phases
1. **Pretraining**: Encoder training on puzzle-symbol mappings
2. **Joint Training**: Bidirectional communication with early stopping based on Generalization Efficiency Score (GES)
3. **Consolidation**: Identification and removal of "recessive" symbols
4. **Remedial Training**: Targeted training on poorly performing puzzles
5. **Addition**: Vocabulary expansion with new puzzles and symbols

### Technical Innovations
- **Adaptive Symbol Consolidation**: Automatic removal of underperforming communication symbols
- **Background Decoder Training**: Concurrent reconstruction training for symbol grounding
- **Multi-Head Attention Pooling**: Learned attention patterns for visual-symbolic mapping
- **Generalization Efficiency Score**: Novel metric for evaluating communication effectiveness

## Usage

### Basic Training Run

```bash
python train_selection.py
```

### Start-Up Web Client

```bash
python web-app.py
```

## Configuration

Create a `config.json` file to customize training parameters:

```json
{
    "max_global_phases": 50,
    "first_pretrain_epochs": 100,
    "pretrain_epochs": 50,
    "initial_puzzle_count": 4,
    "initial_comm_symbols": 4,
    "first_training_cycles": 50,
    "training_cycles": 25,
    "repetitions_per_puzzle": 1,
    "num_distractors": 3,
    "phase_change_indicator": "ges",
    "learning_rate": 7e-7,
    "embedding_dim": 512,
    "hidden_dim": 1024,
    "output_dir": "./outputs"
}
```

## Research Findings

### Emergent Topological Structure
The learned embedding space forms a continuous topological map where different puzzle types exist in structured relationships. Rather than discrete symbol-to-puzzle mappings, agents discover underlying mathematical relationships between visual patterns.

### Generalization Capabilities
Agents demonstrate genuine pattern recognition, successfully communicating about novel puzzles through:
- **Unseen Communication Testing**: Selection tasks on puzzles never used in training
- **Novel Symbol Induction**: Dynamic symbol creation for completely new puzzle types

### Symbol Consolidation Dynamics
The progressive removal of "recessive" symbols forces agents to develop more robust and generalizable representations, leading to emergent compositional understanding.

## Evaluation Metrics

- **Selection Accuracy**: Success rate in sender→receiver communication tasks
- **Generalization Efficiency Score (GES)**: Normalized metric accounting for vocabulary size and task complexity
- **Symbol Utilization**: Analysis of which symbols become dominant vs. recessive
- **Novel Induction Success**: Performance on completely unseen puzzle types

## Outputs and Visualization

The system generates comprehensive outputs:
- Real-time training visualizations (loss, accuracy, GES, symbol counts)
- Phase-based training plots with moving averages
- Detailed consolidation analysis with visual debugging
- Extensive logging of unseen and novel symbol testing

## Research Applications

This framework provides insights into:
- Emergence of symbolic communication from perceptual grounding
- Training dynamics in multi-agent systems
- Relationship between attention mechanisms and symbol grounding
- Compositional generalization in neural networks

## Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA support recommended for training
- 8GB+ RAM for default configurations
- ARC dataset (arc-agi_test_challenges.json)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{emergent-communication-2025,
  title={Emergent Communication and Symbol Grounding in Visual Reasoning},
  author={[Your Name]},
  year={2024},
  url={https://github.com/DBravy/symbolic-emergence}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for improvements or extensions to the research framework.

## Contact

For questions about the research or technical implementation, please open an issue or contact [dbrayjr5@gmail.com].
