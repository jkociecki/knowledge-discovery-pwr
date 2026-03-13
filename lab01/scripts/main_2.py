import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import Player, BoardManager


def main():
    print(f"Running script: {os.path.basename(__file__)}\n")

    players_data = (("Jedrek", 10, 3), ("Ania", 8, 4), ("Bot_1", 5, 2))

    players_list = [Player(n, p, s) for n, p, s in players_data]

    board = BoardManager(players_list)

    resources = board.get_available_resources()
    winners = board.check_victory(10)

    print(f"Available resources on board: {resources}")

    if winners:
        print(f"winner(s): {', '.join(winners)}! GG WP.")
    else:
        print("The game continues...")


if __name__ == "__main__":
    main()
