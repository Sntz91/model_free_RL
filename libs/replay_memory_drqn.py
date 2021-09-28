import numpy as np
from typing import Tuple, List, Union
from collections import namedtuple, deque

Experience = namedtuple("Experience",
                           field_names = ["observation", "action", "reward"])

class ReplayMemory:
    """
    Replay Memory for Recurrent Neural Networks
    """
    def __init__(self, capacity: int, n_sequence_length: int):
        self.buffer = deque(maxlen=capacity)
        self.n_sequence_length = n_sequence_length
  

    def __len__(self):
        return len(self.buffer)
    
    
    def append(self, exp_sequence):
        #if episode not long enough (n_sequence_length), then reject it
        if(len(exp_sequence) > self.n_sequence_length):
            self.buffer.append(exp_sequence)
        
        
    def sample_episode(self):
        idx = np.random.choice(len(self))
        return self.buffer[idx]
        
        
    def sample(self, batch_size: int = 1) -> Tuple:
        #So a batch now contains batch_size *  sequences, right?
        
        trajectories = []
        for _ in range(batch_size):
            episode = self.sample_episode()

            start_idx = np.random.choice(len(episode)-self.n_sequence_length)
            trajectory = episode[start_idx : (start_idx + self.n_sequence_length)]
            trajectories.append(trajectory)
        
        return trajectories



    
if __name__ == '__main__':     
    rm = ReplayMemory(10, 3)
    episode = 0

    #Episode
    while(True):
        episode+=1
        print("Start episode %d" % (episode))
        done = False
        episode_timestep = 0
        
        #Timestep
        while(not done):
            episode_timestep += 1
            exp_sequence = []
            
            #simulate length of episode
            n_timesteps = np.random.rand() * 100
            n_timesteps = int(n_timesteps) + 1
            
            #simulate episode
            for i in range(n_timesteps):
                exp_sequence.append(Experience(i, i+2, i))
                
            done = True
        rm.append(exp_sequence)
            
        if(episode>=10):
            break
    
    print(rm.sample(2)[0])