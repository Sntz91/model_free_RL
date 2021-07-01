from DQN.DQN_Agent import DQN_Agent
from DQN.DDQN_Agent import DDQN_Agent
from DQN.PER_Agent import PER_Agent
from libs.atari_wrappers import make_atari, wrap_deepmind
import gym
import json

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    config = config['config_mnih2013']
    #config = config['test_config']

    env_name = 'SeaquestNoFrameskip-v4'
    #env_name = 'BreakoutNoFrameskip-v4'

    #different for different environments!
    env = make_atari(env_name, skip_noop=True, skip_maxskip=False, max_episode_steps=108000)
    env = wrap_deepmind(env, pytorch_img=True, episode_life=True) #episode_life in training, but not in evaluation!

    #dqn = DDQN_Agent(env, env_name, config)
    dqn = PER_Agent(env, env_name, config)
    dqn.train_agent()