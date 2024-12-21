import numpy as np

states = [
    #Case 1
    (2, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 1, 1, 0, 0),

    #Case 2
    (1, 1, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 1, 0), (1, 1, 0, 0, 0, 1, 0, 0),
    (1, 1, 0, 0, 1, 0, 0, 0), (1, 1, 0, 0, 1, 1, 0, 0),
    # (1, 1, 0, 0, 1, 1, 0, 1), DOVREBBE ESSERE IMPOSSIBILE TROVARSI IN QUESTO STATO
    # (1, 1, 0, 0, 1, 1, 1, 0), DOVREBBE ESSERE IMPOSSIBILE TROVARSI IN QUESTO STATO
    (1, 1, 0, 0, 0, 1, 1, 0), # STATO AGGIUNTO
    (1, 1, 0, 0, 1, 0, 0, 1), # STATO AGGIUNTO
    (1, 1, 0, 0, 0, 1, 0, 1), # STATO AGGIUNTO
    (1, 1, 0, 0, 1, 0, 1, 0), # STATO AGGIUNTO

    #Case 3
    (1, 0, 1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 1, 0, 0), (1, 0, 1, 0, 1, 0, 0, 0),

    #Case 4
    (1, 0, 0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 1, 0, 0), (1, 0, 0, 1, 1, 0, 0, 0),

    #Case 5
    (0, 2, 0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0, 0, 1), (0, 2, 0, 0, 0, 0, 1, 0), (0, 2, 0, 0, 0, 0, 1, 1),
    (0, 2, 0, 0, 0, 1, 0, 0), (0, 2, 0, 0, 0, 1, 0, 1), (0, 2, 0, 0, 0, 1, 1, 0), (0, 2, 0, 0, 0, 1, 1, 1),
    (0, 2, 0, 0, 1, 0, 0, 0), (0, 2, 0, 0, 1, 0, 0, 1), (0, 2, 0, 0, 1, 0, 1, 0), (0, 2, 0, 0, 1, 0, 1, 1),
    (0, 2, 0, 0, 1, 1, 0, 0), (0, 2, 0, 0, 1, 1, 0, 1), (0, 2, 0, 0, 1, 1, 1, 0), (0, 2, 0, 0, 1, 1, 1, 1),

    #Case 6
    (0, 1, 1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 1, 0, 0),
    (0, 1, 1, 0, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0, 1, 0),

    #Case 7
    (0, 1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 0, 1), (0, 1, 0, 1, 0, 0, 1, 0), (0, 1, 0, 1, 0, 1, 0, 0),
    (0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0, 1, 0),

    #Case 8
    (0, 0, 1, 1, 0, 0, 0, 0),

    #Case 9
    (0, 0, 2, 0, 0, 0, 0, 0),

    #Case 10
    (0, 0, 0, 2, 0, 0, 0, 0)
]


rewards = [
    #Case 1
    [0, 0], [0, 0],

    #Case 2
    [0, 0], [-4, 2], [4, -4], [-4, 4],
    [4, -4], [-4, 4],
    # [-4, 4],  RICOMPENSE RELATIVE AD UNO STATO IMPOSSIBILE
    # [4, -4],  RICOMPENSE RELATIVE AD UNO STATO IMPOSSIBILE
    [-2, 2],    # RICOMPENSE RELATIVE AD UNO STATO AGGIUNTO
    [2, -2],    # RICOMPENSE RELATIVE AD UNO STATO AGGIUNTO
    [-4, 4],    # RICOMPENSE RELATIVE AD UNO STATO AGGIUNTO
    [4, -4],    # RICOMPENSE RELATIVE AD UNO STATO AGGIUNTO

    #Case 3
    [0, 0], [-4, 4], [4, -4],

    #Case 4
    [0, 0], [-4, 4], [4, -4],

    #Case 5
    [0, 0], [-4, 4], [4, -4], [0, 0],
    [-4, 4], [-4, 4], [-4, 4], [-4, 4],
    [4, -4], [4, -4], [4, -4], [4, -4],
    [0, 0], [-4, 4], [4, -4], [0, 0],

    #Case 6
    [0, 0], [-4, 4], [4, -4], [-4, 4],
    [-4, 4], [4, -4], [4, -4],

    #Case 7
    [0, 0], [0, 4], [4, 0], [0, 4],
    [0, 4], [4, 0], [4, 0],

    #Case 8
    [0, 0],

    #Case 9
    [0, 0],

    #Case 10
    [0, 0]
]


def get_reward_from_strategy(state: dict, action: int, old_player_pos: np.ndarray, move_pieces: np.ndarray) -> int:
    reward = 0

    print(state.values())

    # Return the index of the state.
    index_state = states.index(tuple(state.values()))

    # Return the reward given the state and the action that the agent decided to do.
    reward += rewards[index_state][action]

    if len(move_pieces) == 2:
        if state['HOME'] >= 1:
            return reward
        # If the player move the token that is the nearest to the goal.
        if reward == 0:
            other_action = (action + 1) % 2 # To know the other token.
            if old_player_pos[action] > old_player_pos[other_action]:
                reward += 2
            elif old_player_pos[action] < old_player_pos[other_action]:
                reward -= 2

    return reward