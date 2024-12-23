import ludopy
import numpy as np
import sys


sys.path.append("../")


def randwalk():
    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        print(f"\nPlayer's turn: {player_i}")
        print(f"Dice value: {dice}")
        print(f"Pieces the player can move: {move_pieces}")
        print(f"Player's piece positions: {player_pieces}")
        print(f"Enemy's piece positions: {enemy_pieces}")
        print(f"Did the player win? {'Yes' if player_is_a_winner else 'No'}")
        print(f"Is there a winner in the game? {'Yes' if there_is_a_winner else 'No'}")

        if len(move_pieces):
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # Select a random piece
        else:
            piece_to_move = -1  # No piece to move

        print(f'piece_to_move {piece_to_move}')

        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), next_player = g.answer_observation(piece_to_move)

        print(f"\nPlayer's turn: {player_i}")
        print(f"Dice value: {dice}")
        print(f"Pieces the player can move: {move_pieces}")
        print(f"Player's piece positions: {player_pieces}")
        print(f"Enemy's piece positions: {enemy_pieces}")
        print(f"Did the player win? {'Yes' if player_is_a_winner else 'No'}")
        print(f"Is there a winner in the game? {'Yes' if there_is_a_winner else 'No'}")
        print(f'Next player: {next_player}')

    # Save the game video
    # print("Saving game video")
    # g.save_hist_video("game_video.mp4")

    return True


if __name__ == '__main__':
    # Execute the randwalk function
    randwalk()

