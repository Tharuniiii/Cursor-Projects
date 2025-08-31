# snake_ladder.py

import random

class SnakeLadderGame:
    def __init__(self):
        self.snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
        self.ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
        self.positions = [0, 0]  # Player 1 and Player 2
        self.current_player = 0
        self.winner = None

    def roll_dice(self):
        return random.randint(1, 6)

    def move(self, steps):
        if self.winner is not None:
            return
        pos = self.positions[self.current_player] + steps
        if pos > 100:
            pos = self.positions[self.current_player]  # Can't move
        else:
            pos = self.snakes.get(pos, self.ladders.get(pos, pos))
        self.positions[self.current_player] = pos
        if pos == 100:
            self.winner = self.current_player
        self.current_player = 1 - self.current_player  # Switch player

    def get_positions(self):
        return self.positions

    def get_winner(self):
        return self.winner