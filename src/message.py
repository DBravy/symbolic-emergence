from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    symbols: List[int]
    
    def __post_init__(self):
        # Validate that all symbols are within valid range
        if not all(0 <= symbol < 1000 for symbol in self.symbols):
            raise ValueError("All symbols must be between 0 and 999")