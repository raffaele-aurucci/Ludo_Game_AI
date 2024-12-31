from collections import deque
import random

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen: int, seed: int = None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility.
        if seed is not None:
            random.seed(seed)

    def append(self, transition: tuple) -> None:
        self.memory.append(transition)

    def sample(self, sample_size: int) -> list:
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)