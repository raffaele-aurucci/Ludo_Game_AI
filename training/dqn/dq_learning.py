import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
from tqdm import tqdm

import ludopy
from environment.ludo_environment import LudoEnv
from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import os

from training.utils import draw_wins_plot_over_episodes


SELF_PLAY = True

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

PLOTS_DIR = "../../results/plots"
MODELS_DIR = "../../models"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cpu'

# Deep Q-Learning Agent
class DQLAgent():

    def __init__(self):
        # Hyperparameters
        self.learning_rate_a    = 0.3      # learning rate (alpha)
        self.discount_factor_g  = 0.3      # discount rate (gamma)
        self.network_sync_rate  = 10       # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 100000   # size of replay memory
        self.mini_batch_size    = 32       # size of the training data set sampled from the replay memory
        self.epsilon_init       = 1        # 1 = 100% random actions
        self.epsilon_decay      = 0.995    # epsilon decay rate
        self.epsilon_min        = 0.01     # minimum epsilon value

        # Neural Network
        self.policy_dqn = DQN(state_dim=8, action_dim=2).to(device)
        self.target_dqn = DQN(state_dim=8, action_dim=2).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.memory = ReplayMemory(self.replay_memory_size)

        self.loss_fn = nn.MSELoss()        # NN Loss function.
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)  # NN Optimizer.

        # Path to save info.
        self.LOG_FILE   = os.path.join(LOG_DIR, 'dq_learning.log')
        self.MODEL_FILE = os.path.join(MODELS_DIR, 'dq_learning_agent.pt')
        self.GRAPH_FILE = os.path.join(PLOTS_DIR, 'dq_learning_rewards_per_episode.png')
        self.WIN_FILE = os.path.join(PLOTS_DIR, 'wins_plot_dq_learning.png')

        if SELF_PLAY:
            self.LOG_FILE = os.path.join(LOG_DIR, 'dq_learning_self_play.log')
            self.MODEL_FILE = os.path.join(MODELS_DIR, 'dq_learning_agent_self_play.pt')
            self.GRAPH_FILE = os.path.join(PLOTS_DIR, 'dq_learning_self_play_rewards_per_episode.png')
            self.WIN_FILE = os.path.join(PLOTS_DIR, 'wins_plot_dq_learning_self_play.png')
            self.PRETRAINED_MODEL_FILE = os.path.join(MODELS_DIR, 'dq_learning_agent.pt')

    def run(self, num_of_episodes: int):

        # Time tracking.
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Initialize environment.
        env = LudoEnv(ludopy.Game())

        # History.
        rewards_per_espisode = []
        agent_wins = []
        enemy_wins = []
        epsilon_history = []

        # Initialize epsilon.
        epsilon = self.epsilon_init

        # Track number of steps. Used for syncing policy => target network.
        step_count=0

        # Best reward.
        best_reward = -9999

        if SELF_PLAY:
            self.policy_dqn.load_state_dict(torch.load(self.PRETRAINED_MODEL_FILE))
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            best_reward = 1380.0 # best reward of pretrained model

        # Training.
        for episode in tqdm(range(0, num_of_episodes), desc='Training progress'):

            started_state = [2, 0, 0, 0, 0, 0, 0, 0]

            current_agent_state = torch.tensor(started_state, dtype=torch.float, device=device) # Convert state to tensor directly on device
            current_enemy_state = torch.tensor(started_state, dtype=torch.float, device=device)

            terminated = False
            total_reward = 0.0


            while not terminated:

                # Agent turn (green).
                while not terminated:

                    action = self.choose_action(epsilon, current_agent_state)

                    # Execute action.
                    player_state, enemy_state, info, reward, terminated = env.step(action=action)

                    if info['player_is_a_winner']:
                        agent_wins.append(True)
                        enemy_wins.append(False)

                    new_agent_state = list(player_state.values())

                    # Accumulate rewards.
                    total_reward += reward

                    # Convert new state and reward to tensors on device.
                    new_agent_state = torch.tensor(new_agent_state, dtype=torch.float, device=device)
                    reward = torch.tensor(total_reward, dtype=torch.float, device=device)

                    current_agent_state = new_agent_state

                    # Save experience into memory.
                    self.memory.append((current_agent_state, action, new_agent_state, reward, terminated))

                    # Increment step counter.
                    step_count+=1

                    if terminated:
                        env.reset()
                        rewards_per_espisode.append(total_reward)

                    if info['next_player'] != info['current_player']:
                        break


                # Enemy turn (blue).
                while not terminated:

                    if SELF_PLAY:
                        with torch.no_grad():
                            action = self.policy_dqn(current_enemy_state.unsqueeze(dim=0)).squeeze().argmax()
                    else:
                        action = np.random.randint(0, 2)

                    # Execute action.
                    player_state, enemy_state, info, reward, terminated = env.step(action=action)

                    if info['player_is_a_winner']:
                        agent_wins.append(False)
                        enemy_wins.append(True)

                    new_agent_state = list(player_state.values())
                    new_enemy_state = list(enemy_state.values())

                    new_agent_state = torch.tensor(new_agent_state, dtype=torch.float, device=device)
                    new_enemy_state = torch.tensor(new_enemy_state, dtype=torch.float, device=device)

                    current_enemy_state = new_enemy_state
                    current_agent_state = new_agent_state

                    if terminated:
                        env.reset()
                        rewards_per_espisode.append(total_reward)

                    if info['next_player'] != info['current_player']:
                        break

            # Save model when new best reward is obtained.
            if total_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {total_reward:0.1f} ({(total_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                print(log_message)

                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(self.policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = total_reward

            # Update plot of rewards every x seconds.
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.draw_rewards_per_episode_plot(rewards_per_espisode, epsilon_history)
                last_graph_update_time = current_time

            # If enough experience has been collected optimize the policy network.
            if len(self.memory)>self.mini_batch_size:
                mini_batch = self.memory.sample(self.mini_batch_size)
                self.optimize(mini_batch)

                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0


        if SELF_PLAY:
            draw_wins_plot_over_episodes(num_of_episodes, agent_wins, enemy_wins, self_play=True, path=self.WIN_FILE)
        else:
            draw_wins_plot_over_episodes(num_of_episodes, agent_wins, enemy_wins, self_play=False,path=self.WIN_FILE)

        percentage_win_agent = (sum(agent_wins) / num_of_episodes) * 100
        percentage_win_enemy = (sum(enemy_wins) / num_of_episodes) * 100

        return percentage_win_agent, percentage_win_enemy


    def choose_action(self, epsilon, state):

        action = None

        # Select action based on epsilon-greedy
        if random.random() < epsilon:
            # select random action
            action = np.random.randint(0, 2)
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            # select best action
            with torch.no_grad():
                action = self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

        return action


    def draw_rewards_per_episode_plot(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch):

        # List of experiences.
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors.
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * self.target_dqn(new_states).max(dim=1)[0]

        # Calcuate Q values from current policy
        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters (weights and biases)


if __name__ == '__main__':
    # Repeatability of results.
    if not SELF_PLAY:
        np.random.seed(42)

    dql_agent = DQLAgent()
    percentage_win_agent, percentage_win_enemy = dql_agent.run(num_of_episodes=25000)
    print(f'Percentage win agent: {percentage_win_agent}')
    print(f'Percentage win enemy: {percentage_win_enemy}')
