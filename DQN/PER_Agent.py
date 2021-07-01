import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from libs.PER.replay_memory import ReplayMemory, Experience
import os
from .DQN import DQN, DQN_cartpole
from .Dueling_DQN import DuelingDQN
from libs.PER.weighted_loss import Loss
import resource


class PER_Agent():
    """
        Vanilla DQN Agent from Mnih2013, Mnih2015
    """
    def __init__(self, env, env_name, config):
        #Hyperparameters
        self.hp_replay_memory_capacity = config['hp_replay_memory_capacity']
        self.hp_gamma = config['hp_gamma']
        self.hp_epsilon_start = config['hp_epsilon_start']
        self.hp_epsilon_end = config['hp_epsilon_end']
        self.hp_epsilon_decay_last_frame = config['hp_epsilon_decay_last_frame']
        self.hp_learning_rate = config['hp_learning_rate']
        self.hp_replay_memory_start_after = config['hp_replay_memory_start_after']
        self.hp_batch_size = config['hp_batch_size']
        self.hp_target_update_after = config['hp_target_update_after']
        self.hp_update_frequency = config['hp_update_frequency']

        #Training loop: Episode length differs in each game, so timesteps / frames are better! (bellemare 2017)
        self.nr_of_total_frames = config['nr_of_total_frames'] 
        self.nr_of_evaluation_frames = config['nr_of_evaluation_frames']
        self.nr_of_frames_before_evaluation = config['nr_of_frames_before_evaluation']
        
        #Evaluation variables
        self.timesteps_overall = -1
        self.timesteps_after_last_episode = 0
        self.train_obtained_returns = []
        self.train_avg_returns = []
        self.eval_obtained_returns = []
        self.eval_counter = 0
        self.eval_epsilon = 0.05
        self.eval_state_samples = []

        self.env = env
        self.env_name = env_name
           
        self.replay_memory = ReplayMemory(self.hp_replay_memory_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.training_start_timestamp = datetime.now(tz=None).strftime("%Y-%m-%d_%H-%M-%S")

        #Network
        self.policy_net = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_net = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self._update_target_net()

        self._initialize_evaluation_state_samples_for_max_pred_q_vals(100)


    def _update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def _reset(self):
        """
            reset environment and reset the obtained return
        """
        self.state = self.env.reset()
        self.last_obtained_return = 0.0

    #DDQN! need to tidy up a bit TODO
    def _calc_loss(self, batch):
        """
            Calculate L1-Loss for given batch.
        """
        states, actions, rewards, dones, next_states, is_weights = batch
    
        states_v = torch.from_numpy(states).to(self.device)
        next_states_v = torch.from_numpy(next_states).to(self.device)
        actions_v = torch.from_numpy(actions).to(self.device)
        rewards_v = torch.from_numpy(rewards).to(self.device)
        is_weights_v = torch.from_numpy(is_weights).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        
        state_action_values = self.policy_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states_v).max(1)[1]
            next_state_values = self.target_net(next_states_v).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = (next_state_values * self.hp_gamma) + rewards_v\
            
        #return F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return Loss.weighted_loss(state_action_values, expected_state_action_values, is_weights_v)

    def _get_epsilon(self):
        """
            Get current value for epsilon
        """
        return max(self.hp_epsilon_end, self.hp_epsilon_start - self.timesteps_overall / self.hp_epsilon_decay_last_frame)


    def _select_action(self, eval):
        """
            Select action
        """
        epsilon = self._get_epsilon()
        if eval:
            epsilon = self.eval_epsilon

        with torch.no_grad():
            state = torch.tensor(np.array([self.state], copy=False)).to(self.device)
            q_vals = self.policy_net(state)

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            with torch.no_grad():
                action_ = torch.tensor([action]).to(self.device)
                q_val = q_vals.gather(1, action_.unsqueeze(-1)).squeeze(-1)
        else:
            with torch.no_grad():
                q_val, action = torch.max(q_vals, dim=1)
                action = int(action.item())
                del state

        #TODO TDERROR, 1 pass more through target-net :( The next_state is only known afterwards! 
        return action, q_val #qvals


    def _play_step(self, eval=False):
        """
            Play one step and return if episode ended
        """
        action, state_action_values = self._select_action(eval)
        
        next_state, reward, done, _ = self.env.step(action)

        self.last_obtained_return += reward

        if not eval:
            #calculate q_vals of maxa for next state
            #calculate tderror TODO
            with torch.no_grad():
                next_state_ = torch.tensor(np.array([next_state], copy=False)).to(self.device)
                next_state_values = self.target_net(next_state_).max(1)[0]
                td_error = state_action_values - reward + (self.hp_gamma * next_state_values)
            exp = Experience(self.state, action, reward, done, next_state)
            self.replay_memory.append(exp, td_error.item())

        self.state = next_state
        del next_state
        return done


    def train_agent(self):
        """
            Train the Agent. 
        """
        writer = SummaryWriter(comment="_"+self.env_name)
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hp_learning_rate)
        print ("Start Training on %s" % self.device)

        episode = 0
        while True:
            episode += 1
            self._reset()
            done = False
            ts_episode_started = time.time()

            while(not done):
                self.timesteps_overall += 1
                done = self._play_step()
                
                if len(self.replay_memory) < self.hp_replay_memory_start_after:
                    continue
                    

                #learn
                if self.timesteps_overall % self.hp_update_frequency == 0:
                    optimizer.zero_grad()
                    batch = self.replay_memory.sample(self.hp_batch_size, self.timesteps_overall, self.nr_of_total_frames)
                    loss = self._calc_loss(batch)
                    loss.backward()
                    #gradient clipping, like dueling DQN proposes
                    clipping_value = 10 
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), clipping_value)
                    optimizer.step()

                if self.timesteps_overall % self.hp_target_update_after == 0:
                    self._update_target_net()


            speed = (self.timesteps_overall - self.timesteps_after_last_episode) / (time.time() - ts_episode_started)         
            ram_usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024 / 1024

            self.timesteps_after_last_episode = self.timesteps_overall
            self.train_obtained_returns.append(self.last_obtained_return)
            self.train_avg_returns.append(np.mean(self.train_obtained_returns[-100:]))

            self._write_tensorboard(writer, speed, ram_usage)

            print("Episode %d completed, timesteps played: %d, return: %d, speed %f, epsilon %f" 
                % (episode, self.timesteps_overall, self.last_obtained_return, speed, self._get_epsilon()))
            print("Mean return of last 100 games: %f" % self.train_avg_returns[-1])
            print("Pytorch memory usage: %2f (gb)" % ram_usage)
            print("Size of Replay Memory: %d" % len(self.replay_memory))
            
            #evaluate every nr_of_frames_before_evaluation
            if self.timesteps_overall >= self.nr_of_frames_before_evaluation * (self.eval_counter+1):
                self._evaluate(writer)

            if self.timesteps_overall >= self.nr_of_total_frames:
                break

        writer.close()


    def _initialize_evaluation_state_samples_for_max_pred_q_vals(self, n):
        """
            Get samples for evaluation by sampling n states from random agent
        """
        self.env.reset()
        for i in range(n):
            with torch.no_grad():
                action = self.env.action_space.sample()
                next_state, reward, done, _ =  self.env.step(action)
                self.eval_state_samples.append(next_state)
                del next_state


    def _get_max_pred_q_vals(self):
        """
            Get max q-values for each state in evaluation_state_samples
        """
        with torch.no_grad():
            max_q_vals = []
            for state in self.eval_state_samples:
                state = torch.from_numpy(np.array([state], copy=False)).to(self.device)#.type(self.dtype)
                q_vals = self.policy_net(state)
                max_q_vals.append(q_vals.max().item())
            return np.mean(max_q_vals)


    def _save_model_snapshot(self, score):
        """
            Save model. For each Evaluation Metric, always save the best.
        """ 
        if not os.path.isdir("models/DQN/" + self.env_name + "/" +  self.training_start_timestamp):
            os.mkdir("models/DQN/%s/%s" % (self.env_name, self.training_start_timestamp))
        torch.save(self.policy_net.state_dict(), "models/DQN/%s/%s/snapshot_%d_score_%d.dat" 
                                % (self.env_name, self.training_start_timestamp, self.eval_counter, score))


    def _evaluate(self, writer):
        """
            Pause Training and evaluate agent by running environment like training, 
            but without actually training the model and decreased exploration.
        """
        _eval_iteration_returns = []
        _eval_iteration_max_q_vals = []

        print("***************** Start Evaluation *****************")
        evaluation_episode = 0
        evaluation_frames = 0
        while True:
            evaluation_episode += 1
            self._reset()
            done = False
            while(not done):
                evaluation_frames += 1
                done = self._play_step(eval=True) 

            _eval_iteration_returns.append(self.last_obtained_return)
            _eval_iteration_max_q_vals.append(self._get_max_pred_q_vals())
            print("Evaluation episode %d ended with return %d" %(evaluation_episode, self.last_obtained_return))      

            if evaluation_frames > self.nr_of_evaluation_frames:
                break

        _score_avg_returns = np.mean(_eval_iteration_returns)
        _score_max_q = np.mean(_eval_iteration_max_q_vals)

        print("Avg evaluation score: %f" % _score_avg_returns)
        print("***************** End Evaluation *****************")

        self.eval_counter += 1
        self._save_model_snapshot(_score_avg_returns)
        self.eval_obtained_returns.append(_score_avg_returns)

        writer.add_scalar('Evaluation/AvgTotalReturnPerEpisode', _score_avg_returns, self.eval_counter)
        writer.add_scalar('Evaluation/MaxPredQVals', _score_max_q, self.eval_counter)

    def _write_tensorboard(self, writer, speed, ram_usage):
        writer.add_scalar('Training/AvgTotalReturn', self.train_avg_returns[-1], self.timesteps_overall)
        writer.add_scalar('Training/ObtainedReturns', self.last_obtained_return, self.timesteps_overall)
        writer.add_scalar('Training/MaxPredQVals', self._get_max_pred_q_vals(), self.timesteps_overall)
        writer.add_scalar('Parameter/Epsilon', self._get_epsilon(), self.timesteps_overall)
        writer.add_scalar('Parameter/Speed', speed, self.timesteps_overall)
        writer.add_scalar('Parameter/MemoryUsage', ram_usage, self.timesteps_overall)
        writer.add_scalar('Parameter/ReplayMemorySize', len(self.replay_memory), self.timesteps_overall)
        writer.add_scalar('Parameter/PER_RM_Beta', self.replay_memory.get_beta(), self.timesteps_overall)

   

