import numpy as np

from ludopy.player import HOME_INDEX, enemy_pos_at_pos
from ludopy.player import HOME_AREAL_INDEXS as SAFE_AREAL_INDEXS
from ludopy.player import GOAL_INDEX

TOKEN_1 = 0
TOKEN_2 = 1
START_ENEMY_FROM_PLAYER = 27
BANNED_INDEX = 26
NOT_VALID = -1
PATH = np.arange(1, 52)


def encode_state(old_player_pos: np.ndarray, player_pos: np.ndarray, enemy_pos: np.ndarray) -> dict:
    state = {}

    # Case 1.
    if player_pos[TOKEN_1] == HOME_INDEX and player_pos[TOKEN_2] == HOME_INDEX:
        state = _get_dictionary_of_state(HOME=2, PATH=0, SAFE=0, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="HOME", new_player_pos=player_pos, new_enemy_pos=enemy_pos)

    # Case 2.
    if (player_pos[TOKEN_1] == HOME_INDEX and player_pos[TOKEN_2] in PATH) or \
            (player_pos[TOKEN_2] == HOME_INDEX and player_pos[TOKEN_1] in PATH):
        state = _get_dictionary_of_state(HOME=1, PATH=1, SAFE=0, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="HOME PATH",
                      old_player_pos=old_player_pos, new_player_pos=player_pos, new_enemy_pos=enemy_pos)

    # Case 3.
    if (player_pos[TOKEN_1] == HOME_INDEX and player_pos[TOKEN_2] in SAFE_AREAL_INDEXS) or \
            (player_pos[TOKEN_2] == HOME_INDEX and player_pos[TOKEN_1] in SAFE_AREAL_INDEXS):
        state = _get_dictionary_of_state(HOME=1, PATH=0, SAFE=1, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="HOME", new_player_pos=player_pos, new_enemy_pos=enemy_pos)

    # Case 4.
    if (player_pos[TOKEN_1] == HOME_INDEX and player_pos[TOKEN_2] == GOAL_INDEX) or \
            (player_pos[TOKEN_2] == HOME_INDEX and player_pos[TOKEN_1] == GOAL_INDEX):
        state = _get_dictionary_of_state(HOME=1, PATH=0, SAFE=0, GOAL=1, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="HOME", new_player_pos=player_pos, new_enemy_pos=enemy_pos)

    # Case 5.
    if player_pos[TOKEN_1] in PATH and player_pos[TOKEN_2] in PATH:
        state = _get_dictionary_of_state(HOME=0, PATH=2, SAFE=0, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="PATH", old_player_pos=old_player_pos, new_player_pos=player_pos,
                      new_enemy_pos=enemy_pos)

    # Case 6.
    if (player_pos[TOKEN_1] in PATH and player_pos[TOKEN_2] in SAFE_AREAL_INDEXS) or \
            (player_pos[TOKEN_2] in PATH and player_pos[TOKEN_1] in SAFE_AREAL_INDEXS):
        state = _get_dictionary_of_state(HOME=0, PATH=1, SAFE=1, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="PATH", old_player_pos=old_player_pos, new_player_pos=player_pos,
                      new_enemy_pos=enemy_pos)

    # Case 7.
    if (player_pos[TOKEN_1] in PATH and player_pos[TOKEN_2] == GOAL_INDEX) or \
            (player_pos[TOKEN_2] in PATH and player_pos[TOKEN_1] == GOAL_INDEX):
        state = _get_dictionary_of_state(HOME=0, PATH=1, SAFE=0, GOAL=1, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)
        _update_state(old_state=state, update_state="PATH", old_player_pos=old_player_pos, new_player_pos=player_pos,
                      new_enemy_pos=enemy_pos)

    # Case 8.
    if (player_pos[TOKEN_1] in SAFE_AREAL_INDEXS and player_pos[TOKEN_2] == GOAL_INDEX) or \
            (player_pos[TOKEN_2] in SAFE_AREAL_INDEXS and player_pos[TOKEN_1] == GOAL_INDEX):
        state = _get_dictionary_of_state(HOME=0, PATH=0, SAFE=1, GOAL=1, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)

    # Case 9.
    if player_pos[TOKEN_1] in SAFE_AREAL_INDEXS and player_pos[TOKEN_2] in SAFE_AREAL_INDEXS:
        state = _get_dictionary_of_state(HOME=0, PATH=0, SAFE=2, GOAL=0, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)

    # Case 10.
    if player_pos[TOKEN_1] == GOAL_INDEX and player_pos[TOKEN_2] == GOAL_INDEX:
        state = _get_dictionary_of_state(HOME=0, PATH=0, SAFE=0, GOAL=2, ENEMY_VULNERABLE_TO_TOKEN_1=0,
                                         ENEMY_VULNERABLE_TO_TOKEN_2=0, TOKEN_1_VULNERABLE_TO_ENEMY=0,
                                         TOKEN_2_VULNERABLE_TO_ENEMY=0)

    return state


def _get_dictionary_of_state(HOME: int, PATH: int, SAFE: int, GOAL: int, ENEMY_VULNERABLE_TO_TOKEN_1: int,
                             ENEMY_VULNERABLE_TO_TOKEN_2: int, TOKEN_1_VULNERABLE_TO_ENEMY: int, TOKEN_2_VULNERABLE_TO_ENEMY: int):
    return {
        'HOME': HOME,
        'PATH': PATH,
        'SAFE': SAFE,
        'GOAL': GOAL,
        'ENEMY_VULNERABLE_TO_TOKEN_1': ENEMY_VULNERABLE_TO_TOKEN_1,
        'ENEMY_VULNERABLE_TO_TOKEN_2': ENEMY_VULNERABLE_TO_TOKEN_2,
        'TOKEN_1_VULNERABLE_TO_ENEMY': TOKEN_1_VULNERABLE_TO_ENEMY,
        'TOKEN_2_VULNERABLE_TO_ENEMY': TOKEN_2_VULNERABLE_TO_ENEMY,
    }


def _update_state(old_state: dict, update_state: str, old_player_pos = None, new_player_pos = None, new_enemy_pos = None) -> dict:
    state = old_state

    if update_state.__contains__("HOME"):
        if (new_enemy_pos[TOKEN_1] == START_ENEMY_FROM_PLAYER or new_enemy_pos[TOKEN_2] == START_ENEMY_FROM_PLAYER) \
                and new_player_pos[TOKEN_1] == HOME_INDEX:
            state['ENEMY_VULNERABLE_TO_TOKEN_1'] = 1
        if (new_enemy_pos[TOKEN_1] == START_ENEMY_FROM_PLAYER or new_enemy_pos[TOKEN_2] == START_ENEMY_FROM_PLAYER) \
                and new_player_pos[TOKEN_2] == HOME_INDEX:
            state['ENEMY_VULNERABLE_TO_TOKEN_2'] = 1

    if update_state.__contains__("PATH"):

        # If enemy has eaten the player (in the PATH).
        if old_player_pos[TOKEN_1] > new_player_pos[TOKEN_1] or old_player_pos[TOKEN_2] > new_player_pos[TOKEN_2]:
            if old_state['PATH'] == 1:
                state['PATH'] = 0
                state['HOME'] = old_state['HOME'] + 1
            elif old_state['PATH'] == 2:
                if old_player_pos[TOKEN_1] > new_player_pos[TOKEN_1] and old_player_pos[TOKEN_2] > new_player_pos[TOKEN_2]:
                    state['PATH'] = 0
                    state['HOME'] = 2
                else:
                    state['PATH'] = 1
                    state['HOME'] = 1


        token_1_pos_from_enemy = enemy_pos_at_pos(new_player_pos[TOKEN_1])[1][0]
        token_2_pos_from_enemy = enemy_pos_at_pos(new_player_pos[TOKEN_2])[1][0]

        # If the TOKEN1-TOKEN2 of the player is vulnerable to the TOKEN1-TOKEN2 of the enemy
        # or the enemy is vulnerable to TOKEN1-TOKEN2 of player.
        for TOKEN in [TOKEN_1, TOKEN_2]:
            if new_enemy_pos[TOKEN] in PATH and new_enemy_pos[TOKEN] != BANNED_INDEX:

                if token_1_pos_from_enemy != NOT_VALID:

                    if 1 <= (token_1_pos_from_enemy - new_enemy_pos[TOKEN]) <= 6:
                        # Check for extreme case (player and enemy cannot eat).
                        if not (new_player_pos[TOKEN] >= 27 and new_enemy_pos[TOKEN] >= 47):
                            state['TOKEN_1_VULNERABLE_TO_ENEMY'] = 1

                    elif -6 <= (token_1_pos_from_enemy - new_enemy_pos[TOKEN]) <= -1:
                        # Same.
                        if not (new_player_pos[TOKEN] >= 47 and new_enemy_pos[TOKEN] >= 27):
                            state['ENEMY_VULNERABLE_TO_TOKEN_1'] = 1

                if token_2_pos_from_enemy != NOT_VALID:

                    if 1 <= (token_2_pos_from_enemy - new_enemy_pos[TOKEN]) <= 6:
                        # Same.
                        if not (new_player_pos[TOKEN] >= 27 and new_enemy_pos[TOKEN] >= 47):
                            state['TOKEN_2_VULNERABLE_TO_ENEMY'] = 1

                    elif -6 <= (token_2_pos_from_enemy - new_enemy_pos[TOKEN]) <= -1:
                        # Same.
                        if not (new_player_pos[TOKEN] >= 47 and new_enemy_pos[TOKEN] >= 27):
                            state['ENEMY_VULNERABLE_TO_TOKEN_2'] = 1

            # If the TOKEN1-TOKEN2 of enemy is in HOME and the TOKEN1-TOKEN2 of the player are on its START_INDEX.
            if new_enemy_pos[TOKEN] == HOME_INDEX:
                if new_player_pos[TOKEN_1] == START_ENEMY_FROM_PLAYER:
                    state['TOKEN_1_VULNERABLE_TO_ENEMY'] = 1
                if new_player_pos[TOKEN_2] == START_ENEMY_FROM_PLAYER:
                    state['TOKEN_2_VULNERABLE_TO_ENEMY'] = 1


    return state








