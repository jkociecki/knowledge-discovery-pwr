from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class Player:
    nickname: str
    points: int
    settlements: int


class BoardManager:
    def __init__(self, players: List[Player]):
        self.players = players
        self.tiles = {2: "Ore", 6: "Brick", 8: "Lumber", 12: "Wool"}

    def get_available_resources(self) -> Set[str]:
        return {res for res in self.tiles.values()}

    def check_victory(self, win_condition: int):
        is_winner = lambda p: p.points >= win_condition
        return [p.nickname for p in self.players if is_winner(p)]
