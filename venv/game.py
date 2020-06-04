import pygame
import snake
import random
import learning

class Game:
  def __init__(self, surface_size=(800, 800), cell_size=(40, 40), speed=30, display_training=False):
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
    self.mdp = learning.Mdp(self.board_size)
    self.rl = learning.RLearning(self.mdp, episodes=100)
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
    print("Episodes", self.rl.episodes, "Alpha", self.rl.alpha, "Gamma", self.rl.gamma)
    steps = 0
    while self.game_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.game_running = False

      self.rl.step()
      steps += 1
      if self.display_training or self.episode == self.rl.episodes:
        self.refresh()

      ret = self.handle_terminal(steps)
      if ret == "game_over":
        steps = 0
        break
      elif ret == "next_episode":
        steps = 0
        continue

      if self.handle_goal():
        steps = 0

      self.keyboard_handle() # for debuging

    pygame.quit()

  def slow_down_last_episode(self):
    if self.episode > self.rl.episodes - 1:
      self.speed = 10

  def handle_terminal(self, steps):
    if self.mdp.is_terminal(self.rl.state) or steps >= self.rl.max_steps:
      print(f"Ep {self.episode}.    {steps}/{self.rl.max_steps} steps    "
            f"score {self.mdp.snake.eaten_fruits}    epsilon {self.rl.epsilon:.3}")
      print(f"Weights::  O {self.rl.weights[0]:.3} | R {self.rl.weights[1]:.3} "
            f"| G {self.rl.weights[2]:.3} | W {self.rl.weights[3]:.3}")

      self.episode += 1
      self.rl.epsilon -= self.rl.epsilon_delta
      self.rl.epsilon = self.rl.epsilon if self.rl.epsilon > 0.0 else 0.0

      if self.episode > self.rl.episodes:
        self.game_running = False
        return "game_over"
      else:
        self.reset()
        return "next_episode"
    return "play"

  def handle_goal(self):
    if self.mdp.is_goal(self.rl.state):
      self.mdp.rand_fruit_coords()
      self.mdp.snake.eaten_fruits += 1
      head = self.mdp.snake.move()
      self.mdp.cells[head[0]][head[1]] = "S"
      return True
    else:
      head = self.mdp.snake.move()
      tail = self.mdp.snake.pop_tail()
      self.mdp.cells[tail[0]][tail[1]] = "."
      self.mdp.cells[head[0]][head[1]] = "S"
      return False

  def refresh(self):
    self.slow_down_last_episode()

    self.board.fill(self.board_color)
    self.draw_fruit()
    self.draw_snake()
    self.screen.blit(self.board, self.cell_size)

    self.side_bar.fill(self.background_color)
    score_text = self.font.render(f"Score: {self.mdp.snake.eaten_fruits}", 1, self.font_color)
    episode_text = self.font.render(f"Episode: {self.episode}/{self.rl.episodes}", 1, self.font_color)
    alpha_text = self.font.render(f"Alpha: {self.rl.alpha:.2f}", 1, self.font_color)
    epsilon_text = self.font.render(f"Epsilon: {self.rl.epsilon:.2f}", 1, self.font_color)
    train_text = self.font.render(f"Train: {self.rl.train}", 1, self.font_color)
    self.side_bar.blit(score_text, (40, 40))
    self.side_bar.blit(episode_text, (40, 120))
    self.side_bar.blit(alpha_text, (40, 200))
    self.side_bar.blit(epsilon_text, (40, 280))
    self.side_bar.blit(train_text, (40, 360))
    self.screen.blit(self.side_bar, self.side_bar_size)

    pygame.display.update()
    self.clock.tick(self.speed)

  def draw_fruit(self):
    mapped_coords = self.map_coords(self.mdp.fruit_coords)
    mapped_coords = (mapped_coords[0] + self.cell_size[0] // 2, mapped_coords[1] + self.cell_size[1] // 2)
    pygame.draw.circle(self.board, self.fruit_color, mapped_coords, self.cell_size[0] // 2)

  def draw_snake(self):
    for i in range(len(self.mdp.snake.body)):
      mapped_coords = self.map_coords(self.mdp.snake.body[i])
      (r, g, b) = self.snake_color
      r = (r + i*5) if (r + i*5) < 255 else 255
      segment_color = (r, g, b)
      pygame.draw.rect(self.board, segment_color, mapped_coords + self.cell_size)

  def reset(self):
    self.mdp.reset()
    self.rl.state = self.mdp.create_state(self.mdp.snake.body[0], "Right")

  def keyboard_handle(self):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
      return (0, -1)
    elif keys[pygame.K_RIGHT]:
      return (0, 1)
    elif keys[pygame.K_DOWN]:
      return (1, 0)
    elif keys[pygame.K_UP]:
      return (-1, 0)
    elif keys[pygame.K_d]:
      print("Debug")
      return None
    else:
      return None