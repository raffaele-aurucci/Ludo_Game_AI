import numpy as np

TOTAL_NUMBER_OF_TAILES = 58
DICE_MOVE_OUT_OF_HOME = 6
NO_ENEMY = -1

# Numeric constants used as markers for cells.
TAILE_FREE = 0
TAILE_HOME = 1
TAILE_START = 2
TAILE_GOAL_AREAL = 3
TAILE_GOAL = 4

NULL_POS = -1
HOME_INDEX = 0
START_INDEX = 1

HOME_AREAL_INDEXS = [52, 53, 54, 55, 56]
GOAL_INDEX = 57

# Safe area for specific tails.
ENEMY_1_GLOB_INDX = 14
ENEMY_2_GLOB_INDX = 27
ENEMY_3_GLOB_INDX = 40

# Set all the tails to 0.
BORD_TILES = np.full(TOTAL_NUMBER_OF_TAILES, TAILE_FREE)

# Initialization board.
BORD_TILES[HOME_INDEX] = TAILE_HOME
BORD_TILES[START_INDEX] = TAILE_START
BORD_TILES[HOME_AREAL_INDEXS] = TAILE_GOAL_AREAL
BORD_TILES[GOAL_INDEX] = TAILE_GOAL

# Number of tails between the enemy home and the player home.
ENEMY_1_INDX_AT_HOME = 40
ENEMY_2_INDX_AT_HOME = 27
ENEMY_3_INDX_AT_HOME = 14


def enemy_pos_at_pos(pos):
    """
    Returns the index's the other players has to be in to be in the same location as the one given in pos

    :param pos: The location to check for
    :type pos: int
    :return enemy_pos: The locations the enemy's pieces has to be at
    :rtype enemy_pos: list of list
    """
    enemy_pos = []

    for enemy_start_pos, enemy_pos_at_start in [[ENEMY_1_GLOB_INDX, ENEMY_1_INDX_AT_HOME],
                                                [ENEMY_2_GLOB_INDX, ENEMY_2_INDX_AT_HOME],
                                                [ENEMY_3_GLOB_INDX, ENEMY_3_INDX_AT_HOME]]:
        post_offset = enemy_start_pos - 1   # 13
        pre_offset = enemy_pos_at_start - 1 # 39

        if pos == enemy_start_pos:
            pos_enemy = [START_INDEX, HOME_AREAL_INDEXS[0]] # (1, 51)
        elif pos < 0:
            pos_enemy = [max(enemy_pos_at_start + pos, -1)] # bho
        elif START_INDEX <= pos < enemy_start_pos: # between 1 - 13
            pos_enemy = [pos + pre_offset]
        elif pos > HOME_AREAL_INDEXS[0] or pos == HOME_INDEX:
            pos_enemy = [-1]
        else:
            pos_enemy = [pos - post_offset] # between 14 - 38 (green side)
        enemy_pos.append(pos_enemy)

    return enemy_pos


# Check if there are enemies at the input pos.
def get_enemy_at_pos(pos, enemies):
    """
    Returns the enemy's and the pieces they have at the given location

    :param pos: The location to check for
    :type pos: int
    :param enemies: The locations for the enemy's pieces in a list of 4 lists

    :returns:
    - enemy_at_pos: The enemy's there are at the location
    - enemy_pieces_at_pos: The pieces the enemy's has at the location
    :rtype enemy_at_pos: list
    :rtype enemy_pieces_at_pos: list of list

    """
    # Get the pos the enemy's has to be at to be at the same pos.
    other_enemy_pos_at_pos = enemy_pos_at_pos(pos)
    # Check if there is a enemy and how many pieces the enemy has there.
    enemy_at_pos = NO_ENEMY
    enemy_pieces_at_pos = []

    for enemy_i, other_enemy_pos in enumerate(other_enemy_pos_at_pos):
        # Check if there already is found a enemy at pos.
        if enemy_at_pos != NO_ENEMY:
            # If there is then stop checking for more (there can only be one).
            break

        for o_pos in other_enemy_pos:
            if o_pos == NULL_POS:
                continue

            for enemy_pice, enemy_pos in enumerate(enemies[enemy_i]):
                if enemy_pos == o_pos:
                    enemy_pieces_at_pos.append(enemy_pice)
                    enemy_at_pos = enemy_i

    # enemy_at_pos = (NO_ENEMY / (1,2,3)); enemy_pieces_at_pos = indexes of enemy.
    return enemy_at_pos, enemy_pieces_at_pos


class Player:
    """
    A class used by the Game class. This class is not needed for normal use
    """

    def __init__(self):
        """
        Makes a player with 2 pieces at the home locations
        """
        self.pieces = []
        self.number_of_pieces = 2
        self.set_all_pieces_to_home() # Set all the pieces to 0.

    def get_pieces_that_can_move(self, dice):
        """
        Return the pieces that can move with the given dice

        :param dice: The dice the move will be done with
        :type dice: int
        :return: movable_pieces: A list with the pieces that can be moved
        :rtype movable_pieces: list

        """
        movable_pieces = []
        # Go though all the pieces.
        for piece_i, piece_place in enumerate(self.pieces):
            # If the piece is a goal then the piece can't move.
            if BORD_TILES[piece_place] == TAILE_GOAL: # The piece has arrived at the goal.
                continue

            # If the piece is at home and the dice is DICE_MOVE_OUT_OF_HOME then the dice can move out of the home place.
            elif BORD_TILES[piece_place] == TAILE_HOME and dice == DICE_MOVE_OUT_OF_HOME:
                movable_pieces.append(piece_i)
            # If the piece is not at home or at the goal it can move.
            elif BORD_TILES[piece_place] != TAILE_HOME:
                movable_pieces.append(piece_i)
        return movable_pieces

    def player_winner(self):
        """
        Returns rather the player is a winner or not

        :return: winner: A bool that indicate rather the player is a winner or not
        :rtype winner: bool
        """
        # Go though all the pieces.
        for piece_place in self.pieces:
            # If a piece is not at the goal is not the winner.
            if BORD_TILES[piece_place] != TAILE_GOAL:
                return False
        # If no piece was not at the goal the player is the winner.
        return True

    def set_pieces(self, pieces):
        """
        Sets the players pieces

        :param pieces: The pieces to set the players pieces to
        """
        self.pieces = np.copy(pieces)

    def get_pieces(self):
        """
        Returns the players pieces

        :return pieces: The players pieces
        :rtype pieces: list
        """
        return np.copy(self.pieces)

    def move_piece(self, piece, dice, enemies):
        """
        Move the players piece the given dice following the game rules. Returns the new locations of the enemy's pieces

        :param piece: The piece to move
        :type piece: int
        :param dice: The dice to make the move with
        :type dice: int
        :param enemies: The enemy's pieces
        :type enemies: list with 4 lists each with 4 int's
        :return enemies: The new locations of the enemy's pieces
        :rtype enemies: list with 4 lists each with 4 int's

        """
        enemies_new = enemies.copy()
        old_piece_pos = self.pieces[piece]
        new_piece_pos = old_piece_pos + dice

        move_enemy_home_from_poss = []
        enemy_at_pos, enemy_pieces_at_pos = get_enemy_at_pos(new_piece_pos, enemies)


        # If the dice is 0 then no movement can be done.
        if dice == 0:
            pass

        # At goal.
        elif BORD_TILES[old_piece_pos] == TAILE_GOAL:
            # The piece can not move.
            pass

        # Goal areal.
        elif BORD_TILES[old_piece_pos] == TAILE_GOAL_AREAL:
            if new_piece_pos <= GOAL_INDEX:
                self.pieces[piece] = new_piece_pos
            else: # If the piece exceeds the size of the board, he stays still.
                self.pieces[piece] = old_piece_pos

        # If the piece is in the home and the dice value is 6.
        # The Home areal.
        elif BORD_TILES[old_piece_pos] == TAILE_HOME:
            if dice == DICE_MOVE_OUT_OF_HOME:
                self.pieces[piece] = START_INDEX

                # Set the enemy there might be at START_INDEX to moved.
                move_enemy_home_from_poss.append(START_INDEX) # The enemies came back at their home.

        # If the token moves in the tile, it eats the opponent who occupies it.
        elif BORD_TILES[old_piece_pos] == TAILE_FREE or \
                BORD_TILES[new_piece_pos] == TAILE_FREE:
            if enemy_at_pos != NO_ENEMY:
                move_enemy_home_from_poss.append(new_piece_pos)
            self.pieces[piece] = new_piece_pos

        # If the case was not caught then there is an error.
        else:
            print("\nold_piece_pos:", old_piece_pos, "\nnew_piece_pos", new_piece_pos,
                  "\nBORD_TILES[old_piece_pos]:", BORD_TILES[old_piece_pos],
                  "\nBORD_TILES[new_piece_pos]:", BORD_TILES[new_piece_pos], "\ndice:", dice)
            raise RuntimeError("The new_piece_pos case was not handel")

        # Check if there is any enemy there has to be moved.
        if len(move_enemy_home_from_poss):
            # Go through the pos where enemy has to be moved from.
            for pos in move_enemy_home_from_poss:
                # Get the enemy at the pos.
                enemy_at_pos, enemy_pieces_at_pos = get_enemy_at_pos(pos, enemies)
                # Check if there was an enemy at the pos.
                if enemy_at_pos != NO_ENEMY:
                    # If there is only one enemy then move the enemy home.
                    if len(enemy_pieces_at_pos) == 1:
                        for enemy_piece in enemy_pieces_at_pos:
                            enemies_new[enemy_at_pos][enemy_piece] = HOME_INDEX # There is only one enemy and HE came back home.
                    # If there is more than one then move own piece home.
                    else:
                        self.pieces[piece] = new_piece_pos + 1 # There is more than one enemy piece, in this case the player go on 1 step.

        return enemies_new # New position of enemies.

    def set_all_pieces_to_home(self):
        """
        Sets all the players pieces to the home index
        """
        self.pieces = []
        for i in range(self.number_of_pieces):
            self.pieces.append(HOME_INDEX)

