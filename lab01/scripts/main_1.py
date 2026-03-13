import os
import random


def main():
    print(f"Running script: {os.path.basename(__file__)}\n")
    dice_rolls = [random.randint(1, 6) + random.randint(1, 6) for _ in range(10)]

    stats = {}

    for roll in dice_rolls:
        if roll == 7:
            print(f"Roll {roll}: Thief is moving!")
        elif roll in [6, 8]:
            print(f"Roll {roll}: High probability field hit!")
        else:
            print(f"Roll {roll}: Common production.")

        stats[roll] = stats.get(roll, 0) + 1

    print(f"\nDice Statistics: {stats}")


if __name__ == "__main__":
    main()
