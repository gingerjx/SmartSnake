import game

if __name__ == '__main__':
  """It is highly not recommended to train snake on large boards"""
  board_size = {"32x32": (680, 680), "26x26": (560, 560), "20x20": (440, 440), "14x14": (320, 320), "8x8": (200, 200)}
  game = game.Game(board_size["8x8"], train=True, episodes=2000, alpha=0.01)
  #game = game.Game(board_size["8x8"], train=False)
  print("Game started")
  game.start()
  game.pygame_quit()
  print("Game over")

