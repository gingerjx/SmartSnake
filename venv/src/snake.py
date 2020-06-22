import pygame

class Snake:
  def __init__(self, init_coords, board_size):
    self.eaten_fruits = 0
    self.board_size = board_size
    self.direction = (0, -1)  # (0 1) right, (0 -1) left, (-1 0) up, (1 0) down
    self.body = [init_coords[0], init_coords[1]]

  def get_head(self):
    """Return coords of snakes' head"""
    return self.body[0]

  def get_tail(self):
    """Return coords of snakes' tail"""
    return self.body[-1]

  def is_crushed(self, coords):
    """Check if given 'coords' intersecting with snakes' body, excluding head"""
    for i in range(1, len(self.body)):
      if self.body[i] == coords:
        return True
    return False

  def move(self):
    """Move snake - based on current direction insert new coords at the beginning of the body.
       Return these new coords"""
    (x, y) = self.body[0]
    self.body.insert(0, (x + self.direction[0], y + self.direction[1]))
    return self.body[0]

  def pop_tail(self):
    """Return tail coords and delete them."""
    tail = self.body.pop()
    return tail
