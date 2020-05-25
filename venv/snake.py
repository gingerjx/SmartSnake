import pygame

class Snake:
  def __init__(self, head_coords, board_size):
    self.eaten_fruits = 0
    self.board_size = board_size
    self.direction = (0, 1)  # (1 0) right, (-1 0) left, (0 -1) up, (0 1) down
    self.body = [head_coords, (head_coords[0]-1, head_coords[1])]

  def backward_move(self):
    return (self.direction[0] * -1, self.direction[1] * -1)

  def wall_crash(self):
    (head_x, head_y) = self.body[0]
    (board_width, board_height) = self.board_size
    if head_x < 0 or head_x >= board_width or head_y < 0 or head_y >= board_height:
        return True
    else:
        return False

  def self_crash(self):
    for i in range(1, len(self.body)):
      if self.body[i] == self.body[0]:
        return True
    return False

  def check_crash(self, coords):
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

  def reset(self, head_coords):
    self.eaten_fruits = 0
    self.direction = (0, 1)
    self.body = [head_coords, (head_coords[0] - 1, head_coords[1])]

