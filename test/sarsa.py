import pickle
import numpy as np
from tqdm import tqdm

import ludopy
from environment.ludo_environment import LudoEnv
from environment.rewards import states

SAVE_VIDEO = True
saved = False

def execute(mode:str = 'S-VS-RANDOM'):

    # Initialize environment.
    env = LudoEnv(ludopy.Game())

    Q_agent = None
    Q_enemy = None

    global saved

    if mode == 'S-VS-RANDOM':
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_agent = pickle.load(file)

    elif mode == 'S-VS-S':
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_agent = pickle.load(file)
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_enemy = pickle.load(file)

    elif mode == 'S_SELF-VS-RANDOM':
        with open('../models/sarsa_agent_self_play.pkl', 'rb') as file:
            Q_agent = pickle.load(file)

    elif mode == 'S_SELF-VS-S':
        with open('../models/sarsa_agent_self_play.pkl', 'rb') as file:
            Q_agent = pickle.load(file)
        with open('../models/sarsa_agent.pkl', 'rb') as file:
            Q_enemy = pickle.load(file)

    elif mode == 'S_SELF-VS-S_SELF':
        with open('../models/sarsa_agent_self_play.pkl', 'rb') as file:
            Q_agent = pickle.load(file)
        with open('../models/sarsa_agent_self_play.pkl', 'rb') as file:
            Q_enemy = pickle.load(file)

    else:
        raise ValueError('Invalid mode.')


    terminated = False

    # Respect to Q-table (agent).
    current_agent_state_idx = 0

    # Respect to Q-table (enemy).
    current_enemy_state_idx = 0

    # Bool value.
    win_agent = None


    while not terminated:

        # Player turn (green).
        while not terminated:

            # Choose the action to run
            action = int(np.argmax(Q_agent[current_agent_state_idx, :]))

            player_state, enemy_state, info, reward, terminated = env.step(action=action)

            # Return the index of the state.
            new_agent_state_idx = states.index(tuple(player_state.values()))

            # Update current state.
            current_agent_state_idx = new_agent_state_idx

            # Agent win (green).
            if terminated:
                win_agent = True
                if SAVE_VIDEO and not saved:
                    print('')
                    env.render(mode=mode)
                    saved = True
                env.reset()

            if info['next_player'] != info['current_player']:
                break


        # Enemy turn (blue).
        while not terminated:

            action = None

            # If you play against yourself, the action chosen is not random.
            if not mode.__contains__('RANDOM'):
                action = int(np.argmax(Q_enemy[current_enemy_state_idx, :]))
            else:
                action = np.random.randint(0, 2)

            # Player_state is always green.
            player_state, enemy_state, info, reward_enemy, terminated = env.step(action=action)

            # Return the index of the state.
            current_agent_state_idx = states.index(tuple(player_state.values()))

            if not mode.__contains__('RANDOM'):

                # Return the index of the state.
                new_enemy_state_idx = states.index(tuple(enemy_state.values()))

                # Update current state.
                current_enemy_state_idx = new_enemy_state_idx

            # Enemy win (blue).
            if terminated:
                win_agent = False
                if SAVE_VIDEO and not saved:
                    print('')
                    env.render(mode=mode)
                    saved = True
                env.reset()

            if info['next_player'] != info['current_player']:
                break

    return win_agent


if __name__ == '__main__':
    np.random.seed(42)
    num_wins_agent = 0

    mode = 'S_SELF-VS-S'

    for i in tqdm(range(0, 5000), desc='Game progress'):
        win_agent = execute(mode)
        if win_agent:
            num_wins_agent += 1

    percentage_win_agent = (num_wins_agent / 5000) * 100
    print(f'\nMode: {mode}')
    print(f'Percentage win agent: {percentage_win_agent}')
    print(f'Percentage win enemy: {100 - percentage_win_agent}')

