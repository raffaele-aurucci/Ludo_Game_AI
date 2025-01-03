import numpy as np
from matplotlib import pyplot as plt


def draw_wins_plot_over_episodes(num_of_episodes: int, agent_wins: list, enemy_wins: list, file_path: str):
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
    ax.legend(loc='upper left')

    plt.savefig(file_path)