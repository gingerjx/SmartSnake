import pygame

class Snake:
  def __init__(self, init_coords, board_size):
    self.eaten_fruits = 0
    self.board_size = board_size
    self.direction = (0, -1)  # (0 1) right, (0 -1) left, (-1 0) up, (1 0) down
    self.body = [init_coords[0], init_coords[1]]

  def head(self):
    return self.body[0]

  def crush(self, coords):
    for i in range(1, len(self.body)):
      if self.body[i] == coords:
        return True
    return False

  def move(self):
    (x, y) = self.body[0]
    self.body.insert(0, (x + self.direction[0], y + self.direction[1]))
    return self.body[0]

  def pop_tail(self):
    tail = self.body.pop()
    return tail

  def reset(self, init_coords):
    self.eaten_fruits = 0
    self.body = [init_coords[0], init_coords[1]]
    self.direction = (0, -1)

