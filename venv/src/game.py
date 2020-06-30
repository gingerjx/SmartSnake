import pygame
import snake
from learning import Agent, MDP
from collections import deque
import numpy as np

class Game:
  def __init__(self, surface_size=(800, 800), train=False, episodes=100, alpha=0.001, gamma=0.9, epsilon=0.8, accuracy=0.2, speed=15):
    self.fruit_color = (46, 200, 50)
    self.board_color = self.font_color = (204, 209, 209)
    self.background_color = (33, 47, 61)
    self.snake_color = (24, 26, 70)
    self.surface_size = (surface_size[0], surface_size[1] + 75)
    self.cell_size = (30, 30)
    self.speed = speed

    self.side_size = (surface_size[0], 75)
    self.board_size_window = (surface_size[0] - 2 * self.cell_size[0], surface_size[1] - 2 * self.cell_size[1])
    self.board_size = (self.board_size_window[1] // self.cell_size[0], self.board_size_window[1] // self.cell_size[1])
    self.fruit_radius = self.cell_size[0] // 2

    self.game_running = True
    self.agent = Agent(MDP(self.board_size), train, episodes, gamma, alpha, epsilon, self.board_size[0]*self.board_size[1])
    self.episode = 0
    self.accuracy = accuracy

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption('SmartSnake' + str(self.board_size))
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode(self.surface_size)
    self.screen.fill(self.background_color)
    self.board = pygame.Surface(self.board_size_window)
    self.side_bar_pos = (0, surface_size[1])
    self.side_bar = pygame.Surface(self.side_size)
    self.side_bar.fill(self.background_color)
    self.screen.blit(self.side_bar, self.side_bar_pos)
    self.font = pygame.font.SysFont("comicsans", self.cell_size[0])

  def map_coords(self, coords):
    """Return mapped 'coords' to pixel coords"""
    return (coords[1] * self.cell_size[1], coords[0] * self.cell_size[0])

  def start(self):
    """Start game, contain game loop"""
    steps = 0
    last_scores = deque([0, 0, 0, 0], maxlen=4)
    while self.game_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.game_running = False
      """Display snake if it is last episode or flag is set"""
      if not self.agent.train:
        self.refresh(steps)
      """Make a reinforcement learning step"""
      self.agent.step()
      steps += 1
      """Check if it's end of episode and handle it"""
      if self.agent.is_terminal() or steps >= self.agent.max_steps:
        last_scores.append(self.agent.get_score())
        if self.is_game_over(last_scores, steps):
          self.game_running = False
          break
        else:
          self.reset(steps)
          steps = 0
          continue
      """Reset steps if agent reached fruit"""
      if self.agent.is_goal():
        steps = 0
      """Make snake move"""
      self.agent.mdp.move_snake()

    if self.agent.train:
      self.agent.save_model()

  def is_game_over(self, last_scores, steps):
    print(f"Ep. {self.episode}    steps {steps}   score {self.agent.get_score()}    epsilon {self.agent.epsilon}")
    size = self.board_size[0]*self.board_size[1]
    if np.mean(last_scores)/size >= self.accuracy or self.episode >= self.agent.episodes - 1:
      return True
    else: False

  def reset(self, steps):
    """"Reset episode"""
    self.episode += 1
    self.agent.reset_episode()

  def refresh(self, steps):
    """Refresh application view"""
    self.board.fill(self.board_color)
    self.draw_fruit()
    self.draw_snake()
    self.screen.blit(self.board, self.cell_size)

    self.side_bar.fill(self.background_color)
    score_text = self.font.render(f"Score: {self.agent.mdp.snake.eaten_fruits}", 1, self.font_color)
    steps_text = self.font.render(f"Steps: {steps}/{self.agent.max_steps}", 1, self.font_color)

    padding = self.cell_size[0]
    center = self.surface_size[0] // 2
    self.side_bar.blit(score_text, (center - 40, 0))
    self.side_bar.blit(steps_text, (center - 50, padding))
    self.screen.blit(self.side_bar, self.side_bar_pos)

    pygame.display.update()
    self.clock.tick(self.speed)

  def draw_fruit(self):
    """Draw fruit on screen"""
    mapped_coords = self.map_coords(self.agent.mdp.fruit_coords)
    mapped_coords = (mapped_coords[0] + self.cell_size[0] // 2, mapped_coords[1] + self.cell_size[1] // 2)
    pygame.draw.circle(self.board, self.fruit_color, mapped_coords, self.cell_size[0] // 2)

  def draw_snake(self):
    """Draw snake on screen"""
    for i in range(len(self.agent.mdp.snake.body)):
      mapped_coords = self.map_coords(self.agent.mdp.snake.body[i])
      (r, g, b) = self.snake_color
      r = (r + i*5) if (r + i*5) < 255 else 255
      segment_color = (r, g, b)
      pygame.draw.rect(self.board, segment_color, mapped_coords + self.cell_size)

  def pygame_quit(self):
    pygame.quit()
