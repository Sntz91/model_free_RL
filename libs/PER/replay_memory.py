from typing import Tuple, List, Union
from collections import namedtuple, deque
import numpy as np
from .SumTree import SumTree

Experience = namedtuple("Experience", 
                            field_names = ["state", "action", "reward", "done", "next_state"])

class ReplayMemory:
    """
    Original Replay Memory by Lin. Used for vanilla DQN, no prioritized Replay or bootstrapping with n>1.
    Used to store and sample experiences
    """
    def __init__(self, capacity: int, epsilon = 0.01, alpha = 0.6, beta = 0.4) -> None:
        """
        Args: 
            capacity: size of buffer
        """
        self.buffer = deque(maxlen=capacity)
        self.sumTree = SumTree([0 for x in range(capacity)])
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.first_append = True

    def __len__(self):
        return len(self.buffer)

    def append(self, sample: Experience, tderror) -> None:
        """
        Append sample
        Args:
            sample: A sample of an experience to store. Experience is a tuple(state, action, reward, done, next_state)
        """
        self.buffer.append(sample)
        priority = self.get_priority(tderror)
        self.sumTree.update(len(self.buffer)-1, priority)


    def get_priority(self, tderror):
        pi = abs(tderror) + self.epsilon
        if self.first_append:
            Pi = (pi ** self.alpha) / pi
            self.first_append = False
        else:
            Pi = (pi ** self.alpha) / self.sumTree.top_node.value
        return Pi


    def _recalculate_beta(self, frame, max_frame):
        m = frame/max_frame
        rmax = 1
        rmin = 1/max_frame
        tmin = self.beta_start
        tmax = 1
        self.beta = ((m - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin

    def get_beta(self):
        return self.beta

    def sample(self, batch_size, frame: int, max_frame: int) -> Tuple:
        """
        Return batch of buffer, randomly (uniformely).
        Args: 
            batch_size: size of batch
        """
        self._recalculate_beta(frame, max_frame) 
        idxs = []
        is_weights = []
        #TODO: beta linearly increasing over time!
        for _ in range(batch_size):
            idx = self.sumTree.draw_idx()
            idxs.append(idx)
            wi = ((1 / len(self.buffer)) * (1 / self.sumTree.leaf_nodes[idx].value ))**self.beta #hoch -beta? TODO
            is_weights.append(wi)


        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in idxs])


        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), \
            np.array(next_states), np.array(is_weights, dtype=np.float32)

    
