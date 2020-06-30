from game import Game

if __name__ == '__main__':
  """It is highly not recommended to train snake on large boards"""
  board_size = {"24x24": (800, 800), "20x20": (680, 680), "16x16": (560, 560), "12x12": (440, 440), "8x8": (320, 320)}
  train = False
  episodes = 500
  alpha = 0.0005
  gamma = 0.9
  epsilon = 0.8
  accuracy = 0.25
  speed = 30

  print("Train: " + str(train))
  if train:
    print("Episodes: " + str(episodes))
    print("Alpha: " + str(alpha))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("Accuracy: " + str(accuracy))
  else:
    print("Speed: " + str(speed))

  game = Game(board_size["12x12"], train, episodes, alpha, gamma, epsilon, accuracy, speed)
  print("Game started")
  game.start()
  game.pygame_quit()
  print("Game over")

