import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import ludopy
from environment.ludo_environment import LudoEnv
from environment.rewards import states


SAVE_CSV_RESULTS = False     # True in grid search mode, else False.
SAVE_PLOTS = True            # True when not in grid search mode, else False.
SELF_PLAY = True


def training_episodes(num_of_episodes: int, exploration_prob: float, learning_rate: float, discount_factor: float) -> tuple:

    agent_wins = []
    enemy_wins = []
    total_rewards = []

    # Initialize environment.
    env = LudoEnv(ludopy.Game())

    # Random if not self-play.
    Q_agent = np.random.rand(53, 2)

    if SELF_PLAY:
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_agent = pickle.load(file)

    # Using in self-play.
    Q_enemy = None

    if SELF_PLAY:
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_enemy = pickle.load(file)

    for episode in range(num_of_episodes):
        terminated = False

        # Respect to Q-table (player).
        current_agent_state_idx = 0

        # Respect to Q-table (enemy).
        current_enemy_state_idx = 0

        total_reward = 0

        while not terminated:

            # Agent turn (green).
            while not terminated:

                # Choose the action from Q-table.
                action = choose_action(Q_agent, current_agent_state_idx, exploration_prob)

                player_state, enemy_state, info, reward, terminated = env.step(action=action)

                if info['player_is_a_winner']:
                    agent_wins.append(True)
                    enemy_wins.append(False)

                # Return the index of the state.
                new_agent_state_idx = states.index(tuple(player_state.values()))

                # Obtain new action.
                new_action = choose_action(Q_agent, new_agent_state_idx, exploration_prob)

                # Update Q table.
                update_Q_table(Q_agent, current_agent_state_idx, new_agent_state_idx, reward, action, new_action, learning_rate, discount_factor)

                # Update current state.
                current_agent_state_idx = new_agent_state_idx

                total_reward += reward

                # Agent win (green).
                if terminated:
                    env.reset()
                    total_rewards.append(total_reward)

                if info['next_player'] != info['current_player']:
                    break


            # Enemy turn (blue).
            while not terminated:

                action = None

                # If you play against yourself, the action chosen is not random.
                if SELF_PLAY:
                    action = int(np.argmax(Q_enemy[current_enemy_state_idx, :]))
                else:
                    action = np.random.randint(0, 2)

                # Player_state is always green.
                player_state, enemy_state, info, _, terminated = env.step(action=action)

                if info['player_is_a_winner']:
                    agent_wins.append(False)
                    enemy_wins.append(True)

                # Return the index of the state.
                current_agent_state_idx = states.index(tuple(player_state.values()))

                if SELF_PLAY:

                    # Return the index of the state.
                    new_enemy_state_idx = states.index(tuple(enemy_state.values()))

                    # Update current state.
                    current_enemy_state_idx = new_enemy_state_idx

                # Enemy win (blue).
                if terminated:
                    env.reset()
                    total_rewards.append(total_reward)

                if info['next_player'] != info['current_player']:
                    break


        exploration_prob = max(0.01, exploration_prob * 0.995)


    # Plot wins over episodes.
    if SAVE_PLOTS:
        draw_plot(num_of_episodes, agent_wins, enemy_wins)

    percentage_win_agent = (sum(agent_wins) / num_of_episodes) * 100
    percentage_win_enemy = (sum(enemy_wins) / num_of_episodes) * 100

    return Q_agent, percentage_win_agent, percentage_win_enemy


def choose_action(Q: np.ndarray, state_idx: int, exploration_prob: float) -> int:
    if np.random.rand() < exploration_prob:
        return np.random.randint(0, 2)  # Random exploration.
    else:
        return int(np.argmax(Q[state_idx, :]))


def update_Q_table(Q: np.ndarray, current_player_state_idx: int, new_player_state_idx: int, reward:int, action: int,
                   new_action: int, learning_rate: float, discount_factor: float):

    Q[current_player_state_idx, action] += learning_rate * (reward + discount_factor * Q[new_player_state_idx, new_action]
                                                            - Q[current_player_state_idx, action])


def draw_plot(num_of_episodes: int, agent_wins: list, enemy_wins: list):

    agent_cumsum = np.cumsum(agent_wins)  # Cumulative sum for agent
    enemy_cumsum = np.cumsum(enemy_wins)  # Cumulative sum for enemy

    agent_percentage = (agent_cumsum / num_of_episodes) * 100
    enemy_percentage = (enemy_cumsum / num_of_episodes) * 100

    fig, ax = plt.subplots()

    ax.plot(agent_percentage, label='Agent Wins %', color='forestgreen')
    ax.plot(enemy_percentage, label='Enemy Wins %', color='steelblue')
    ax.set_title('Wins Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Percentage Win')
    ax.legend()


    if SELF_PLAY:
        plt.savefig('../results/plots/wins_plot_sarsa_self_play.png')
    else:
        plt.savefig('../results/plots/wins_plot_sarsa.png')

    plt.show()


if __name__ == '__main__':
    num_episodes = 5000

    # Uncomment this part for grid search.
    # exploration_probabilities = [0.1, 0.2, 0.3]
    # learning_rates = [0.3, 0.4, 0.5]
    # discount_factors = [0.3, 0.5, 0.7, 0.9]

    # The best configuration.
    exploration_probabilities = [0.1]
    learning_rates = [0.3]
    discount_factors = [0.9]

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

    for exploration_prob, learning_rate, discount_factor in tqdm(param_combinations, desc="Training Progress",
                                                                 total=len(param_combinations)):

        Q, percentage_win_agent, percentage_win_enemy = training_episodes(num_episodes, exploration_prob, learning_rate,
                                                                          discount_factor)

        results.append([num_episodes, exploration_prob, learning_rate, discount_factor,
                        round(percentage_win_agent, 2), round(percentage_win_enemy, 2)])

        if percentage_win_agent > best_percentage_win_agent:
            best_percentage_win_agent = percentage_win_agent

            if SELF_PLAY:
                with open('../models/sarsa_agent_self_play.pkl', 'wb') as file:
                    pickle.dump(Q, file)
                print("\nSarsa model saved successfully.")
            else:
                with open('../models/sarsa_agent.pkl', 'wb') as file:
                    pickle.dump(Q, file)
                print("\nSarsa model saved successfully.")

            if not SAVE_CSV_RESULTS:
                print(f'Percentage win agent: {percentage_win_agent}')
                print(f'Percentage win enemy: {percentage_win_enemy}')


    df = pd.DataFrame(results,
                      columns=['Num Episodes', 'Exploration Probability', 'Learning Rate', 'Discount Factor',
                               'Percentage Win Agent', 'Percentage Win Enemy'])

    if SAVE_CSV_RESULTS:
        if SELF_PLAY:
            df.to_csv('../results/training_results_sarsa_self_play.csv', index=False)
        else:
            df.to_csv('../results/training_results_sarsa.csv', index=False)