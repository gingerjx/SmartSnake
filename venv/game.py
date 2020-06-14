import pygame
import snake
from learning import Agent, MDP

class Game:
  def __init__(self, surface_size=(800, 800), cell_size=(40, 40), speed=15, display_training=False):
    self.fruit_color = (46, 200, 50)
    self.board_color = self.font_color = (204, 209, 209)
    self.background_color = (33, 47, 61)
    self.snake_color = (24, 26, 70)
    self.surface_size = (surface_size[0] + 400, surface_size[1])
    self.cell_size = cell_size
    self.speed = speed
    self.display_training = display_training

    self.side_size = (400, surface_size[1])
    self.board_size_window = (surface_size[0] - 2 * cell_size[0], surface_size[1] - 2 * cell_size[1])
    self.board_size = (self.board_size_window[1] // cell_size[0], self.board_size_window[1] // cell_size[1])
    self.fruit_radius = cell_size[0] // 2

    self.game_running = True
    self.agent = Agent(MDP(self.board_size))
    self.episode = 0

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption('SmartSnake')
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode(self.surface_size)
    self.screen.fill(self.background_color)
    self.board = pygame.Surface(self.board_size_window)
    self.side_bar_size = (surface_size[0], 0)
    self.side_bar = pygame.Surface(self.side_size)
    self.side_bar.fill(self.background_color)
    self.screen.blit(self.side_bar, self.side_bar_size)
    self.font = pygame.font.SysFont("comicsans", cell_size[0] + 10)

  def map_coords(self, coords):
    return (coords[1] * self.cell_size[1], coords[0] * self.cell_size[0])

  def start(self):
    steps = 0
    while self.game_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.game_running = False

      if self.display_training or self.episode == self.agent.episodes:
        self.refresh()

      self.agent.step()
      steps += 1
      if self.agent.is_terminal() or steps >= self.agent.max_steps:
        if self.episode > self.agent.episodes:
          self.game_running = False
          break
        else:
          self.reset(steps)
          steps = 0
          continue

      if self.agent.is_goal():
        steps = 0
      self.agent.mdp.move_snake()

    pygame.quit()

  def reset(self, steps):
    print(f"Ep. {self.episode}    steps {steps}   score {self.agent.get_score()}    epsilon {self.agent.epsilon}")
    self.episode += 1
    self.agent.reset_episode()

  def refresh(self):
    self.board.fill(self.board_color)
    self.draw_fruit()
    self.draw_snake()
    self.screen.blit(self.board, self.cell_size)

    self.side_bar.fill(self.background_color)
    score_text = self.font.render(f"Score: {self.agent.mdp.snake.eaten_fruits}", 1, self.font_color)
    episode_text = self.font.render(f"Episode: {self.episode}/{self.agent.episodes}", 1, self.font_color)
    alpha_text = self.font.render(f"Alpha: {self.agent.alpha:.4f}", 1, self.font_color)
    epsilon_text = self.font.render(f"Epsilon: {self.agent.epsilon:.2f}", 1, self.font_color)
    train_text = self.font.render(f"Train: {self.agent.train}", 1, self.font_color)
    self.side_bar.blit(score_text, (40, 40))
    self.side_bar.blit(episode_text, (40, 120))
    self.side_bar.blit(alpha_text, (40, 200))
    self.side_bar.blit(epsilon_text, (40, 280))
    self.side_bar.blit(train_text, (40, 360))
    self.screen.blit(self.side_bar, self.side_bar_size)

    pygame.display.update()
    self.clock.tick(self.speed)

  def draw_fruit(self):
    mapped_coords = self.map_coords(self.agent.mdp.fruit_coords)
    mapped_coords = (mapped_coords[0] + self.cell_size[0] // 2, mapped_coords[1] + self.cell_size[1] // 2)
    pygame.draw.circle(self.board, self.fruit_color, mapped_coords, self.cell_size[0] // 2)

  def draw_snake(self):
    for i in range(len(self.agent.mdp.snake.body)):
      mapped_coords = self.map_coords(self.agent.mdp.snake.body[i])
      (r, g, b) = self.snake_color
      r = (r + i*5) if (r + i*5) < 255 else 255
      segment_color = (r, g, b)
      pygame.draw.rect(self.board, segment_color, mapped_coords + self.cell_size)