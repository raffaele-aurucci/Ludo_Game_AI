import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

import ludopy
from environment.ludo_environment import LudoEnv
from environment.rewards import states


def training_episodes(num_of_episodes: int, exploration_prob: float, learning_rate: float, discount_factor: float, self_play = False) -> tuple:

    agent_wins = []
    enemy_wins = []
    total_rewards_player = []
    total_rewards_enemy = []

    # Initialize environment.
    env = LudoEnv(ludopy.Game())

    # Initialize Q table.
    Q_player = np.random.rand(53, 2)

    # TODO: understand how to do self-play
    Q_enemy = np.random.rand(53, 2)

    for episode in range(num_of_episodes):
        terminated = False

        # Respect to Q-table (player).
        current_player_state_idx = 0

        # Respect to Q-table (enemy).
        current_enemy_state_idx = 0

        total_reward_player = 0
        total_reward_enemy = 0

        while not terminated:

            # Player turn (green).
            while not terminated:

                # Choose the action to run
                action = choose_action(Q_player, current_player_state_idx, exploration_prob)

                player_state, enemy_state, info, reward, terminated = env.step(action=action)

                if info['player_is_a_winner']:
                    agent_wins.append(True)
                    enemy_wins.append(False)

                # Return the index of the state.
                new_player_state_idx = states.index(tuple(player_state.values()))

                # Update Q table.
                updateQTable(Q_player, current_player_state_idx, new_player_state_idx, reward, action, learning_rate, discount_factor)

                # Update current state.
                current_player_state_idx = new_player_state_idx

                total_reward_player += reward

                # Player win (green).
                if terminated:
                    env.reset()
                    total_rewards_player.append(total_reward_player)
                    if self_play:
                        total_rewards_enemy.append(total_reward_enemy)

                if info['next_player'] != info['current_player']:
                    break


            # Enemy turn (blue).
            while not terminated:

                action = None

                # If you play against yourself, the action chosen is not random.
                if self_play:
                    action = choose_action(Q_enemy, current_enemy_state_idx, exploration_prob)
                else:
                    action = np.random.randint(0, 2)

                # Player_state is always green.
                player_state, enemy_state, info, reward_enemy, terminated = env.step(action=action)

                if info['player_is_a_winner']:
                    agent_wins.append(False)
                    enemy_wins.append(True)

                # Return the index of the state.
                current_player_state_idx = states.index(tuple(player_state.values()))

                if self_play:

                    # Return the index of the state.
                    new_enemy_state_idx = states.index(tuple(enemy_state.values()))

                    # Update Q table.
                    updateQTable(Q_enemy, current_enemy_state_idx, new_enemy_state_idx, reward_enemy, action, learning_rate,
                                 discount_factor)

                    # Update current state.
                    current_enemy_state_idx = new_enemy_state_idx

                    total_reward_enemy += reward_enemy


                # Enemy win (blue).
                if terminated:
                    env.reset()
                    total_rewards_player.append(total_reward_enemy)
                    if self_play:
                        total_rewards_enemy.append(total_reward_enemy)

                if info['next_player'] != info['current_player']:
                    break


        exploration_prob = max(0.01, exploration_prob * 0.995)

    # TODO: add plot for win_rate on number of episodes

    percentage_win_agent = (sum(agent_wins) / num_of_episodes) * 100
    percentage_win_enemy = (sum(enemy_wins) / num_of_episodes) * 100

    return Q_player, percentage_win_agent, percentage_win_enemy


def choose_action(Q, state_idx: int, exploration_prob: float) -> int:
    if np.random.rand() < exploration_prob:
        return np.random.randint(0, 2)  # Random exploration.
    else:
        return int(np.argmax(Q[state_idx, :]))


def updateQTable(Q, current_player_state_idx: int, new_player_state_idx: int, reward:int, action: int,
                 learning_rate: float, discount_factor: float):
    Q[current_player_state_idx, action] = (1 - learning_rate) * Q[current_player_state_idx, action] + \
                               learning_rate * (reward + discount_factor * np.max(Q[new_player_state_idx, :]))


if __name__ == '__main__':
    num_episodes = 5000

    exploration_probabilities = [0.1, 0.2, 0.3]
    learning_rates = [0.3, 0.4, 0.5]
    discount_factors = [0.3, 0.5, 0.7, 0.9]

    best_percentage_win_agent = 0

    # Repeatability of results.
    np.random.seed(42)

    # Combination of parameters.
    param_combinations = [
        (exploration_prob, learning_rate, discount_factor)
        for exploration_prob in exploration_probabilities
        for learning_rate in learning_rates
        for discount_factor in discount_factors
    ]

    results = []

    self_play = False

    for exploration_prob, learning_rate, discount_factor in tqdm(param_combinations, desc="Training Progress",
                                                                 total=len(param_combinations)):

        Q, percentage_win_agent, percentage_win_enemy = training_episodes(num_episodes, exploration_prob, learning_rate,
                                                                        discount_factor, self_play=self_play)

        results.append([num_episodes, exploration_prob, learning_rate, discount_factor,
                        round(percentage_win_agent, 2), round(percentage_win_enemy, 2)])

        if percentage_win_agent > best_percentage_win_agent:
            best_percentage_win_agent = percentage_win_agent

            if self_play:
                with open('../models/q_learning_agent_self_play.pkl', 'wb') as file:
                    pickle.dump(Q, file)
                print("\nQ model saved successfully.")
            else:
                with open('../models/q_learning_agent.pkl', 'wb') as file:
                    pickle.dump(Q, file)
                print("\nQ model saved successfully.")

    df = pd.DataFrame(results,
                      columns=['Num Episodes', 'Exploration Probability', 'Learning Rate', 'Discount Factor',
                               'Percentage Win Agent', 'Percentage Win Enemy'])

    if self_play:
        df.to_csv('../results/training_results_q_learning_self_play.csv', index=False)
    else:
        df.to_csv('../results/training_results_q_learning.csv', index=False)