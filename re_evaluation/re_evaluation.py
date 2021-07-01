from DQN.DQN import DQN
import torch
import numpy as np
import gym

class BaseEvaluation():
    pass

class HumanStarts():
    """
        to evaluate agent, run on every 100 starting point sampled from human expert trajectories. Run until end of game or total of 108,000 frames, 
        including the frames the human played until the point. No points from human count. 
    """
    pass

class RandomAgentBaseline():
    """
        Choose random action at 10hz
    """ 
    pass

class NoOpStarts():
    """
        Evaluate agent on 30 episodes of the game it was trained on. Skip random number of frames by repeatedly taking no-op before 
        agent takes over control. Agent plays until end of game or 18,000 frames. Average Score over all 30 episodes.

        No-op max from mnih2015: 30
    """
    def __init__(self, env_name, env, model_path, nr_of_episodes = 30, epsilon = 0.05, random_agent=False):
        self.nr_of_episodes = nr_of_episodes
        self.epsilon = 1
        self.env_name = env_name
        self.env = env 

        if not random_agent:
            self.epsilon = epsilon
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            self.model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device) #TODO: This should be in main file and here just use the model, not the path!
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        self.last_obtained_return = 0
        self.obtained_returns = []
             

    def _get_action(self, state):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.from_numpy(np.array([state], copy=False)).type(self.dtype)
            q_vals = self.model(state)
            _, action = torch.max(q_vals, dim=1)
            action = action.item()
        return action
        



    def evaluate_agent(self):
        self.env = gym.wrappers.Monitor(self.env, 'videos/', force=True)  
        for episode in range(self.nr_of_episodes):
            self.last_obtained_return = 0
            episode_timesteps = 0

            state = self.env.reset()

            done = False
            while(not done):       
                action = self._get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                self.last_obtained_return += reward
                state = next_state

            self.obtained_returns.append(self.last_obtained_return)

        self.env.env.close()
        return [np.mean(self.obtained_returns), np.max(self.obtained_returns)]


    def save_gif(self):
        pass
