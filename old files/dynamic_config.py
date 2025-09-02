"""
Configuration utility for dynamic symbol and sequence length management.
This file makes it easy to adjust plateau detection parameters and symbol/sequence management settings.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class PlateauConfig:
    """Configuration for plateau detection"""
    plateau_cycles: int = 500              # Number of cycles to check for plateau
    plateau_threshold: float = 0.20        # Accuracy range that constitutes plateau (20%)
    min_cycles_before_detection: int = 200 # Minimum cycles before plateau detection starts
    patience_multiplier: float = 1.0       # Multiplier for patience when symbols are added

@dataclass
class SymbolConfig:
    """Configuration for symbol and sequence length management"""
    initial_total_symbols: int = 12        # Starting number of total symbols
    puzzle_symbols: int = 10               # Number of puzzle symbols (fixed)
    max_total_symbols: int = 30            # Maximum total symbols allowed
    initial_seq_length: int = 1            # Starting sequence length
    max_seq_length: int = 10               # Maximum sequence length allowed
    
    @property
    def initial_comm_symbols(self) -> int:
        return self.initial_total_symbols - self.puzzle_symbols
    
    @property
    def max_comm_symbols(self) -> int:
        return self.max_total_symbols - self.puzzle_symbols

@dataclass
class DynamicTrainingConfig:
    """Complete configuration for dynamic training"""
    plateau: PlateauConfig
    symbols: SymbolConfig
    
    # Training parameters
    total_cycles: int = 3000
    learning_rate: float = 1e-3
    
    # Visualization and logging
    plot_frequency: int = 50               # How often to plot metrics
    debug_frequency: int = 100             # How often to show communication debug
    log_file: str = "training_log_dynamic.txt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'plateau': {
                'plateau_cycles': self.plateau.plateau_cycles,
                'plateau_threshold': self.plateau.plateau_threshold,
                'min_cycles_before_detection': self.plateau.min_cycles_before_detection,
                'patience_multiplier': self.plateau.patience_multiplier,
            },
            'symbols': {
                'initial_total_symbols': self.symbols.initial_total_symbols,
                'puzzle_symbols': self.symbols.puzzle_symbols,
                'max_total_symbols': self.symbols.max_total_symbols,
                'initial_seq_length': self.symbols.initial_seq_length,
                'max_seq_length': self.symbols.max_seq_length,
                'initial_comm_symbols': self.symbols.initial_comm_symbols,
                'max_comm_symbols': self.symbols.max_comm_symbols,
            },
            'training': {
                'total_cycles': self.total_cycles,
                'learning_rate': self.learning_rate,
                'plot_frequency': self.plot_frequency,
                'debug_frequency': self.debug_frequency,
                'log_file': self.log_file,
            }
        }
    
    def save_to_json(self, filename: str):
        """Save configuration to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filename: str):
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        plateau_config = PlateauConfig(**data['plateau'])
        symbol_config = SymbolConfig(**data['symbols'])
        
        return cls(
            plateau=plateau_config,
            symbols=symbol_config,
            total_cycles=data['training']['total_cycles'],
            learning_rate=data['training']['learning_rate'],
            plot_frequency=data['training']['plot_frequency'],
            debug_frequency=data['training']['debug_frequency'],
            log_file=data['training']['log_file'],
        )

# Predefined configurations for common scenarios
class PresetConfigs:
    """Predefined configurations for different experimental setups"""
    
    @staticmethod
    def conservative() -> DynamicTrainingConfig:
        """Conservative settings - longer plateau detection, smaller threshold"""
        return DynamicTrainingConfig(
            plateau=PlateauConfig(
                plateau_cycles=1000,           # Wait longer before detecting plateau
                plateau_threshold=0.10,        # Smaller range (10%) = more sensitive
                min_cycles_before_detection=500,
                patience_multiplier=1.5
            ),
            symbols=SymbolConfig(
                initial_total_symbols=12,      # Start with 2 comm symbols
                puzzle_symbols=10,
                max_total_symbols=24,          # More conservative max (7 additions max)
                initial_seq_length=1,
                max_seq_length=8               # Conservative max sequence length
            ),
            total_cycles=5000                  # Longer training
        )
    
    @staticmethod
    def aggressive() -> DynamicTrainingConfig:
        """Aggressive settings - quick plateau detection, larger threshold"""
        return DynamicTrainingConfig(
            plateau=PlateauConfig(
                plateau_cycles=200,            # Quick detection
                plateau_threshold=0.30,        # Larger range (30%) = less sensitive
                min_cycles_before_detection=100,
                patience_multiplier=0.8
            ),
            symbols=SymbolConfig(
                initial_total_symbols=12,
                puzzle_symbols=10,
                max_total_symbols=30,          # Allow more symbols
                initial_seq_length=1,
                max_seq_length=10              # Allow longer sequences
            ),
            total_cycles=3000
        )
    
    @staticmethod
    def research() -> DynamicTrainingConfig:
        """Balanced settings good for research experiments"""
        return DynamicTrainingConfig(
            plateau=PlateauConfig(
                plateau_cycles=500,
                plateau_threshold=0.15,        # 15% range
                min_cycles_before_detection=250,
                patience_multiplier=1.2
            ),
            symbols=SymbolConfig(
                initial_total_symbols=12,
                puzzle_symbols=10,
                max_total_symbols=26,          # 8 communication symbols max
                initial_seq_length=1,
                max_seq_length=8               # Moderate max sequence length
            ),
            total_cycles=4000,
            plot_frequency=25,                 # More frequent plotting
            debug_frequency=50                 # More frequent debugging
        )
    
    @staticmethod
    def minimal_start() -> DynamicTrainingConfig:
        """Start with minimal symbols and grow slowly"""
        return DynamicTrainingConfig(
            plateau=PlateauConfig(
                plateau_cycles=300,
                plateau_threshold=0.25,        # Allow bigger plateaus
                min_cycles_before_detection=200,
                patience_multiplier=1.0
            ),
            symbols=SymbolConfig(
                initial_total_symbols=12,      # Start with 2 comm symbols
                puzzle_symbols=10,
                max_total_symbols=22,          # Conservative growth (6 comm -> 12 comm)
                initial_seq_length=1,
                max_seq_length=7               # Start small, grow to 7
            ),
            total_cycles=6000                  # Longer training needed
        )

def create_custom_config(
    plateau_cycles: int = 500,
    plateau_threshold_percent: float = 20.0,  # Percentage (0-100)
    min_cycles_before_detection: int = 200,
    initial_comm_symbols: int = 2,             # Start with 2 comm symbols
    max_comm_symbols: int = 15,
    initial_seq_length: int = 1,               # Start with sequence length 1
    max_seq_length: int = 8,                   # Max sequence length
    total_training_cycles: int = 3000
) -> DynamicTrainingConfig:
    """
    Create a custom configuration with commonly adjusted parameters.
    
    Args:
        plateau_cycles: Number of cycles to check for plateau
        plateau_threshold_percent: Accuracy range for plateau detection (0-100%)
        min_cycles_before_detection: Minimum cycles before starting plateau detection
        initial_comm_symbols: Number of communication symbols to start with
        max_comm_symbols: Maximum communication symbols allowed
        initial_seq_length: Starting sequence length
        max_seq_length: Maximum sequence length
        total_training_cycles: Total number of training cycles
    
    Returns:
        DynamicTrainingConfig: Custom configuration
    """
    return DynamicTrainingConfig(
        plateau=PlateauConfig(
            plateau_cycles=plateau_cycles,
            plateau_threshold=plateau_threshold_percent / 100.0,  # Convert to 0-1 range
            min_cycles_before_detection=min_cycles_before_detection,
            patience_multiplier=1.0
        ),
        symbols=SymbolConfig(
            initial_total_symbols=10 + initial_comm_symbols,  # 10 puzzle + comm symbols
            puzzle_symbols=10,
            max_total_symbols=10 + max_comm_symbols,
            initial_seq_length=initial_seq_length,
            max_seq_length=max_seq_length
        ),
        total_cycles=total_training_cycles
    )

# Example usage and testing
if __name__ == "__main__":
    # Example: Create and save different configurations
    
    # Conservative configuration
    conservative_config = PresetConfigs.conservative()
    conservative_config.save_to_json("config_conservative.json")
    print("Conservative config:")
    print(f"  Plateau detection: {conservative_config.plateau.plateau_cycles} cycles, {conservative_config.plateau.plateau_threshold:.0%} threshold")
    print(f"  Symbols: {conservative_config.symbols.initial_comm_symbols} -> {conservative_config.symbols.max_comm_symbols} communication symbols")
    print(f"  Sequence: {conservative_config.symbols.initial_seq_length} -> {conservative_config.symbols.max_seq_length} length")
    
    # Custom configuration with the new pattern
    custom_config = create_custom_config(
        plateau_cycles=300,
        plateau_threshold_percent=15.0,
        min_cycles_before_detection=150,
        initial_comm_symbols=2,          # Start with 2
        max_comm_symbols=10,             # Go up to 10 (4 additions: 2->4->6->8->10)
        initial_seq_length=1,            # Start with length 1
        max_seq_length=5,                # Go up to 5 (1->2->3->4->5)
        total_training_cycles=2000
    )
    custom_config.save_to_json("config_custom.json")
    print(f"\nCustom config:")
    print(f"  Plateau detection: {custom_config.plateau.plateau_cycles} cycles, {custom_config.plateau.plateau_threshold:.0%} threshold")
    print(f"  Symbols: {custom_config.symbols.initial_comm_symbols} -> {custom_config.symbols.max_comm_symbols} communication symbols")
    print(f"  Sequence: {custom_config.symbols.initial_seq_length} -> {custom_config.symbols.max_seq_length} length")
    
    # Show how to load a config
    loaded_config = DynamicTrainingConfig.load_from_json("config_custom.json")
    print(f"\nLoaded config matches: {loaded_config.plateau.plateau_cycles == custom_config.plateau.plateau_cycles}")