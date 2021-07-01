from typing import Tuple, List, Union
from collections import namedtuple, deque
import numpy as np

Experience = namedtuple("Experience", 
                            field_names = ["state", "action", "reward", "done", "next_state"])

class ReplayMemory:
    """
    Original Replay Memory by Lin. Used for vanilla DQN, no prioritized Replay or bootstrapping with n>1.
    Used to store and sample experiences
    """
    def __init__(self, capacity: int) -> None:
        """
        Args: 
            capacity: size of buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample: Experience) -> None:
        """
        Append sample
        Args:
            sample: A sample of an experience to store. Experience is a tuple(state, action, reward, done, next_state)
        """
        self.buffer.append(sample)

    def sample(self, batch_size: int = 1) -> Tuple:
        """
        Return batch of buffer, randomly (uniformely).
        Args: 
            batch_size: size of batch
        """
        idxs = np.random.choice(len(self), batch_size, replace=False)

        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), \
            np.array(next_states)

    
