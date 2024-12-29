import numpy as np
import torch
from tqdm import tqdm

import ludopy
from environment.ludo_environment import LudoEnv
from training.dqn.dqn import DQN

SAVE_VIDEO = True
saved = False
device = 'cpu'


def execute(mode:str = 'D-VS-RANDOM'):

    # Initialize environment.
    env = LudoEnv(ludopy.Game())

    policy_dqn_agent = DQN(state_dim=8, action_dim=2).to(device)
    policy_dqn_enemy = DQN(state_dim=8, action_dim=2).to(device)

    global saved

    if mode == 'D-VS-RANDOM':
        policy_dqn_agent.load_state_dict(torch.load('../models/dq_learning_agent.pt'))
        policy_dqn_agent.eval()

    elif mode == 'D-VS-D':
        policy_dqn_agent.load_state_dict(torch.load('../models/dq_learning_agent.pt'))
        policy_dqn_agent.eval()
        policy_dqn_enemy.load_state_dict(torch.load('../models/dq_learning_agent.pt'))
        policy_dqn_enemy.eval()

    elif mode == 'D_SELF-VS-RANDOM':
        policy_dqn_agent.load_state_dict(torch.load('../models/dq_learning_agent_self_play.pt'))
        policy_dqn_agent.eval()

    elif mode == 'D_SELF-VS-D':
        policy_dqn_agent.load_state_dict(torch.load('../models/dq_learning_agent_self_play.pt'))
        policy_dqn_agent.eval()
        policy_dqn_enemy.load_state_dict(torch.load('../models/dq_learning_agent.pt'))
        policy_dqn_enemy.eval()

    elif mode == 'D_SELF-VS-D_SELF':
        policy_dqn_agent.load_state_dict(torch.load('../models/dq_learning_agent_self_play.pt'))
        policy_dqn_agent.eval()
        policy_dqn_enemy.load_state_dict(torch.load('../models/dq_learning_agent.pt'))
        policy_dqn_enemy.eval()

    else:
        raise ValueError('Invalid mode.')


    started_state = [2, 0, 0, 0, 0, 0, 0, 0]
    current_agent_state = torch.tensor(started_state, dtype=torch.float,device=device)
    current_enemy_state = torch.tensor(started_state, dtype=torch.float, device=device)
    terminated = False
    win_agent = None


    while not terminated:

        # Agent turn (green).
        while not terminated:

            action = None

            with torch.no_grad():
                action = policy_dqn_agent(current_agent_state.unsqueeze(dim=0)).squeeze().argmax()

            # Execute action.
            player_state, enemy_state, info, _, terminated = env.step(action=action)

            new_agent_state = list(player_state.values())

            # Convert new state and reward to tensors on device.
            new_agent_state = torch.tensor(new_agent_state, dtype=torch.float, device=device)

            current_agent_state = new_agent_state

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

            if not mode.__contains__('RANDOM'):
                with torch.no_grad():
                    action = policy_dqn_enemy(current_enemy_state.unsqueeze(dim=0)).squeeze().argmax()
            else:
                action = np.random.randint(0, 2)

            # Execute action.
            player_state, enemy_state, info, _, terminated = env.step(action=action)

            new_agent_state = list(player_state.values())
            new_agent_state = torch.tensor(new_agent_state, dtype=torch.float, device=device)
            current_agent_state = new_agent_state

            if not mode.__contains__('RANDOM'):
                new_enemy_state = list(enemy_state.values())
                new_enemy_state = torch.tensor(new_enemy_state, dtype=torch.float, device=device)
                current_enemy_state = new_enemy_state

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

    mode = 'D-VS-RANDOM'

    for i in tqdm(range(0, 5000), desc='Game progress'):
        win_agent = execute(mode)
        if win_agent:
            num_wins_agent += 1

    percentage_win_agent = (num_wins_agent / 5000) * 100
    print(f'\nMode: {mode}')
    print(f'Percentage win agent: {percentage_win_agent}')
    print(f'Percentage win enemy: {100 - percentage_win_agent}')

