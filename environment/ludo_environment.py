import gymnasium as gym
import numpy as np

import ludopy
from environment.rewards import get_reward_from_strategy
from environment.state import encode_state


class LudoEnv(gym.Env):
    def __init__(self, game: ludopy.Game):
        super(LudoEnv, self).__init__()
        self.game = game

        self.observation_space = gym.spaces.Dict(
            {
                'HOME': gym.spaces.Discrete(3),
                'PATH': gym.spaces.Discrete(3),
                'SAFE': gym.spaces.Discrete(3),
                'GOAL': gym.spaces.Discrete(3),
                'ENEMY_VULNERABLE_TO_TOKEN_1': gym.spaces.Discrete(2),
                'ENEMY_VULNERABLE_TO_TOKEN_2': gym.spaces.Discrete(2),
                'TOKEN_1_VULNERABLE_TO_ENEMY': gym.spaces.Discrete(2),
                'TOKEN_2_VULNERABLE_TO_ENEMY': gym.spaces.Discrete(2),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self.current_player = 'green'

        # init observation
        self.player_state = {
            'HOME': 2,
            'PATH': 0,
            'SAFE': 0,
            'GOAL': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_1': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_2': 0,
            'TOKEN_1_VULNERABLE_TO_ENEMY': 0,
            'TOKEN_2_VULNERABLE_TO_ENEMY': 0,
        }

        self.enemy_state = {
            'HOME': 2,
            'PATH': 0,
            'SAFE': 0,
            'GOAL': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_1': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_2': 0,
            'TOKEN_1_VULNERABLE_TO_ENEMY': 0,
            'TOKEN_2_VULNERABLE_TO_ENEMY': 0,
        }

        self.there_is_a_winner = False


    def step(self, action:int) -> tuple:

        reward = 0
        terminated = False

        # check if the game terminated
        if self.there_is_a_winner:
            raise ValueError('The game is terminated, recommended to reset environment.')

        # observe the game
        (dice, move_pieces, old_player_pos, old_enemy_pos, player_is_a_winner,
         there_is_a_winner), player_i = self.game.get_observation()

        # update current player
        self.current_player = player_i

        # choose piece to move
        piece_to_move = -1
        if len(move_pieces):
            if action in move_pieces:
                index = np.where(move_pieces == action)[0][0]
                piece_to_move = move_pieces[index]
            else:
                reward -= 1
                pass # if the agent choose an action that there is not in move_pieces


        # execute move in the game
        (_, _, player_pos, enemy_pos, player_is_a_winner,
         there_is_a_winner), next_player = self.game.answer_observation(piece_to_move)


        # check if the player pass the turn (because has chosen an invalid action)
        if piece_to_move != -1:

            # save old state
            old_player_state = self.player_state
            old_enemy_state = self.enemy_state

            # update state of player and state of enemy
            if self.current_player == 'green':
                reward += get_reward_from_strategy(old_player_state, action, old_player_pos, move_pieces)
                self.player_state = encode_state(old_player_pos, player_pos, enemy_pos)
                self.enemy_state = encode_state(old_enemy_pos, enemy_pos, player_pos)
                reward += self.compute_additional_reward(old_player_state, self.player_state, old_enemy_state, self.enemy_state)
            else: # turn of player blue
                reward += get_reward_from_strategy(old_enemy_state, action, old_enemy_pos, move_pieces)
                self.enemy_state = encode_state(old_player_pos, player_pos, enemy_pos)
                self.player_state = encode_state(old_enemy_pos, enemy_pos, player_pos)
                reward += self.compute_additional_reward(old_enemy_state, self.enemy_state, old_player_state, self.player_state)

            # check if the player is a winner
            if player_is_a_winner:
                reward += 50
                self.there_is_a_winner = True
                terminated = True

        info = {
            'current_player': self.current_player,
            'next_player': next_player,
            'last_dice': dice,
            'old_player_pos': old_player_pos,
            'player_pos': player_pos,
            'old_enemy_pos': old_enemy_pos,
            'enemy_pos': enemy_pos,
            'player_is_a_winner': player_is_a_winner,
        }

        return self.player_state, self.enemy_state, info, reward, terminated


    def compute_additional_reward(self, old_player_state, player_state, old_enemy_state, enemy_state):
        reward = 0

        # Reward when the token of the player enter into PATH from HOME.
        if old_player_state['HOME'] > player_state['HOME']:
            reward += 3

        # Reward when the token of the player enter into safe zone.
        if old_player_state['SAFE'] < player_state['SAFE']:
            reward += 5

        # Reward when at least one token of the player enter into goal tail.
        if old_player_state['GOAL'] < player_state['GOAL']:
            reward += 10

        # Reward when token of player eats the enemy token.
        if old_enemy_state['HOME'] < enemy_state['HOME']:
            reward += 7

        # Reward when the enemy token eats the player token.
        if old_player_state['HOME'] < player_state['HOME']:
            reward -= 7

        return reward

    def reset(self, seed=None, options=None):
        self.game.reset()

        self.current_player = 'green'

        self.there_is_a_winner = False

        # init observation
        self.player_state = {
            'HOME': 2,
            'PATH': 0,
            'SAFE': 0,
            'GOAL': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_1': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_2': 0,
            'TOKEN_1_VULNERABLE_TO_ENEMY': 0,
            'TOKEN_2_VULNERABLE_TO_ENEMY': 0,
        }

        self.enemy_state = {
            'HOME': 2,
            'PATH': 0,
            'SAFE': 0,
            'GOAL': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_1': 0,
            'ENEMY_VULNERABLE_TO_TOKEN_2': 0,
            'TOKEN_1_VULNERABLE_TO_ENEMY': 0,
            'TOKEN_2_VULNERABLE_TO_ENEMY': 0,
        }

    def render(self):
        if self.there_is_a_winner:
            print("Saving game video")
            self.game.save_hist_video("game_video.mp4")


# test of the environment
# ludo_env = LudoEnv(ludopy.Game())
# terminated = False
#
# for i in range(0, 3):
#     while not terminated:
#
#         player_state, enemy_state, info, reward, terminated = ludo_env.step(action=np.random.randint(0,2))
#         print(f"1. GREEN State: {player_state}")
#         print(f"2. BLUE State: {enemy_state}")
#         print(f"3. Info: {info}")
#         print(f"4. Reward: {reward}")
#         print(f"5. Terminated: {terminated}\n")
#
#         if terminated:
#             ludo_env.reset()
#             # break
#
#     terminated = False
#
# ludo_env.render()