import pygame
import snake
import random
import learning as rl

class Game:
  def __init__(self, surface_size=(800, 800), cell_size=(40, 40), speed=8, speed_up=0.1):
    self.fruit_color = (46, 200, 50)
    self.board_color = self.font_color = (204, 209, 209)
    self.background_color = (33, 47, 61)
    self.surface_size = (surface_size[0] + 300, surface_size[1])
    self.cell_size = cell_size
    self.speed = speed
    self.speed_up = speed_up

    self.side_size = (300, surface_size[1])
    self.board_size_window = (surface_size[0] - 2 * cell_size[0], surface_size[1] - 2 * cell_size[1])
    self.board_size = (self.board_size_window[1] // cell_size[0], self.board_size_window[1] // cell_size[1])
    self.cells = [["Empty" for j in range(self.board_size[1])]
                           for i in range(self.board_size[0])]
    self.fruit_radius = cell_size[0] // 2
    self.fruit_coords = (0, 0)  # need to have some initial value to use 'rand_fruit_coords'
    self.fruit_coords = self.rand_fruit_coords()
    self.game_running = True

    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    self.cells[head_coords[0]][head_coords[1]] = "Snake"
    self.cells[head_coords[0] - 1][head_coords[1]] = "Snake"
    self.snake = snake.Snake(head_coords, self.board_size, self.cell_size, snake_color=(24, 26, 70))
    self.mdp = rl.Mdp(self.board_size, self.cells, self.snake, self.fruit_coords)
    self.rl = rl.RLearning(self.mdp)

    pygame.init()
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

  def rand_fruit_coords(self):
    empty_cells = []
    for i in range(self.board_size[0]):
      for j in range(self.board_size[1]):
        if self.cells[i][j] == "Empty":
          empty_cells.append((i, j))
    new_coords = empty_cells[random.randint(0, len(empty_cells)-1)]
    self.cells[self.fruit_coords[0]][self.fruit_coords[1]] = "Empty"
    self.cells[new_coords[0]][new_coords[1]] = "Fruit"
    return new_coords

  def start(self):
    while self.game_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.game_running = False

      new_direction = self.keyboard_handle()
      if new_direction is not None:
        self.snake.direction = new_direction

      head = ()
      if self.snake.ate_fruit(self.fruit_coords):
        self.fruit_coords = self.rand_fruit_coords()
        head = self.snake.move()
        self.speed += self.speed_up
      else:
        head = self.snake.move()
        tail = self.snake.pop_tail()
        self.cells[tail[0]][tail[1]] = "Empty"

      if self.snake.wall_crash() or self.snake.self_crash():
        self.game_running = False
        break
      self.cells[head[0]][head[1]] = "Snake"

      self.refresh()

    pygame.quit()

  def refresh(self):
    self.board.fill(self.board_color)
    self.draw_fruit()
    self.snake.draw(self.board)
    self.screen.blit(self.board, self.cell_size)

    self.side_bar.fill(self.background_color)
    score_text = self.font.render(f"Score: {self.snake.eaten_fruits}", 1, self.font_color)
    speed_text = self.font.render(f"Speed: {self.speed:.2f}", 1, self.font_color)
    self.side_bar.blit(score_text, (60, 40))
    self.side_bar.blit(speed_text, (60, 120))
    self.screen.blit(self.side_bar, self.side_bar_size)

    pygame.display.update()
    self.clock.tick(self.speed)

  def draw_fruit(self):
    mapped_coords = self.map_coords(self.fruit_coords)
    mapped_coords = (mapped_coords[0] + self.cell_size[0] // 2, mapped_coords[1] + self.cell_size[1] // 2)
    pygame.draw.circle(self.board, self.fruit_color, mapped_coords, self.cell_size[0] // 2)

  def keyboard_handle(self):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and self.snake.backward_move() != (-1, 0):
      return (-1, 0)
    elif keys[pygame.K_RIGHT] and self.snake.backward_move() != (1, 0):
      return (1, 0)
    elif keys[pygame.K_DOWN] and self.snake.backward_move() != (0, 1):
      return (0, 1)
    elif keys[pygame.K_UP] and self.snake.backward_move() != (0, -1):
      return (0, -1)
    elif keys[pygame.K_d]:
      return None
    else:
      return None