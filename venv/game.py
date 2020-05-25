import pygame
import snake
import random
import learning as rl

class Game:
  def __init__(self, surface_size=(800, 800), cell_size=(40, 40), speed=10, speed_up=0.1):
    self.fruit_color = (46, 200, 50)
    self.board_color = self.font_color = (204, 209, 209)
    self.background_color = (33, 47, 61)
    self.snake_color = (24, 26, 70)

    self.cell_size = cell_size
    self.surface_size = (surface_size[0] + 300, surface_size[1])
    self.side_size = (300, surface_size[1])
    self.board_size_window = (surface_size[0] - 2 * cell_size[0], surface_size[1] - 2 * cell_size[1])
    self.board_size = (self.board_size_window[1] // cell_size[0], self.board_size_window[1] // cell_size[1])
    self.fruit_radius = cell_size[0] // 2

    self.game_running = True
    self.speed = speed
    self.speed_up = speed_up

    self.rl = rl.RLearning(rl.Mdp(self.board_size))
    self.state = self.rl.mdp.create_state(self.rl.mdp.snake.body[0])

    pygame.display.init()  # can't use pygame.init() cause pygame have some problem with my VB
    pygame.font.init()
    pygame.display.set_caption('SmartSnake')
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode(self.surface_size)
    self.screen.fill(self.background_color)
    self.board = pygame.Surface(self.board_size_window)
    self.side_bar_size = (surface_size[0], 0)
    self.side_bar = pygame.Surface(self.side_size)
    self.side_bar.fill(self.background_color)
    self.screen.blit(self.side_bar,self.side_bar_size)
    self.font = pygame.font.SysFont("comicsans", 50)

  def map_coords(self, coords):
    return (coords[0] * self.cell_size[0], coords[1] * self.cell_size[1])

  def start(self):
    episode = 0
    while self.game_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.game_running = False
          break

      (self.state, action) = self.rl.step(self.state)
      if self.rl.mdp.is_terminal(self.state):
        episode += 1
        if episode >= self.rl.episodes:
          self.game_running = False
          break
        else:
          self.reset()
          continue

      self.rl.mdp.snake.direction = self.rl.mdp.actions[action]

      head = ()
      if self.rl.mdp.is_goal(self.state):
        self.rl.mdp.snake.eaten_fruits += 1
        self.rl.mdp.fruit_coords = self.rl.mdp.rand_fruit_coords()
        head = self.rl.mdp.snake.move()
        self.speed += self.speed_up
      else:
        head = self.rl.mdp.snake.move()
        tail = self.rl.mdp.snake.pop_tail()
        self.rl.mdp.cells[tail[1]][tail[0]] = "."
      self.rl.mdp.cells[head[1]][head[0]] = "S"

      self.refresh()

    pygame.display.quit()

  def reset(self):
    self.rl.epsilon -= self.rl.epsilon_delta
    self.state = self.rl.mdp.reset_mdp()
    (self.state, action) = self.rl.step(self.state)
    self.speed = 10

  def refresh(self):
    self.board.fill(self.board_color)
    self.draw_fruit()
    self.draw_snake()
    self.screen.blit(self.board, self.cell_size)

    self.side_bar.fill(self.background_color)
    score_text = self.font.render(f"Score: {self.rl.mdp.snake.eaten_fruits}", 1, self.font_color)
    speed_text = self.font.render(f"Speed: {self.speed:.2f}", 1, self.font_color)
    self.side_bar.blit(score_text, (60, 40))
    self.side_bar.blit(speed_text, (60, 120))
    self.screen.blit(self.side_bar, self.side_bar_size)

    pygame.display.update()
    self.clock.tick(self.speed)

  def draw_fruit(self):
    mapped_coords = self.map_coords(self.rl.mdp.fruit_coords)
    mapped_coords = (mapped_coords[0] + self.cell_size[0] // 2, mapped_coords[1] + self.cell_size[1] // 2)
    pygame.draw.circle(self.board, self.fruit_color, mapped_coords, self.cell_size[0] // 2)

  def draw_snake(self):
    for i in range(len(self.rl.mdp.snake.body)):
      mapped_coords = self.map_coords(self.rl.mdp.snake.body[i])
      (r, g, b) = self.snake_color
      r = (r + i * 5) if (r + i * 5) < 255 else 255
      segment_color = (r, g, b)
      pygame.draw.rect(self.board, segment_color, mapped_coords + self.cell_size)

  def keyboard_handle(self):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and self.rl.mdp.snake.backward_move() != (-1, 0):
      return (-1, 0)
    elif keys[pygame.K_RIGHT] and self.rl.mdp.snake.backward_move() != (1, 0):
      return (1, 0)
    elif keys[pygame.K_DOWN] and self.rl.mdp.snake.backward_move() != (0, 1):
      return (0, 1)
    elif keys[pygame.K_UP] and self.rl.mdp.snake.backward_move() != (0, -1):
      return (0, -1)
    elif keys[pygame.K_d]:
      return None
    else:
      return None