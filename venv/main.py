import game
import sys

if __name__ == '__main__':
  board_size = {"21x21": (920, 920), "18x18": (800, 800), "15x15": (680, 680), "12x12": (560, 560)}
  game = game.Game(surface_size=board_size["12x12"])
  print("Game started")
  game.start()
  print("Game over")



