import random
import snake

class State:
  def __init__(self, center, obstacles, reward, free_cells):
    self.center = center          # tuple contains head position
    self.obstacles = obstacles    # dictionary contains information about obstacles presence in every four sides
    self.reward = reward          # dictionary contains information about reward presence in every four sides
    self.free_cells = free_cells  # dictionary contains information about percent of free cells in every four sides

  def equals(self, state):
    return  self.center == state.center and\
            self.obstacles == state.obstacles and\
            self.reward == state.reward and \
            self.free_cells == state.free_cells

  # temporary function
  def action_value(self, action):
    obstacles_value = -1.0 if self.obstacles[action] else 0.0
    reward_value = 0.2 if self.reward[action] else 0.0
    # missing free cels
    return obstacles_value + reward_value

class Mdp:
  def __init__(self, board_size):
    self.board_size = board_size
    self.cells = [["." for j in range(self.board_size[1])]
                           for i in range(self.board_size[0])]
    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    self.cells[head_coords[0]][head_coords[1]] = "S"
    self.cells[head_coords[0] - 1][head_coords[1]] = "S"
    self.fruit_coords = (0, 0)  # need to have some initial value to use 'rand_fruit_coords'
    self.fruit_coords = self.rand_fruit_coords()

    self.snake = snake.Snake(head_coords, self.board_size)
    self.actions = {"Right": (0, 1), "Left": (0, -1), "Down": (1, 0), "Up": (-1, 0)}
    self.directions = {(0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"}

  def valid_coords(self, coords):
    (x, y) = coords
    return (x >= 0 and x < self.board_size[0]) and (y >= 0 and y < self.board_size[1])

  def create_state(self, coords):
    (x, y) = coords

    obstacles = {}
    for action in self.actions:
      (dir_x, dir_y) = self.actions[action]
      next_coords = (x + dir_x, y + dir_y)
      if not self.valid_coords(next_coords) or self.cells[next_coords[0]][next_coords[1]] == "S":
        obstacles[action] = True
      else:
        obstacles[action] = False

    reward = {}
    (fruit_x, fruit_y) = self.fruit_coords
    reward["Right"] = True if y < fruit_y else False
    reward["Left"] = True if y > fruit_y else False
    reward["Down"] = True if x < fruit_x else False
    reward["Up"] = True if x > fruit_x else False

    free_cells = self.free_cells(coords)
    return State(coords, obstacles, reward, free_cells)

  def free_cells(self, coords):
    # returns empty_cells/square_area ratio for each direction (free means fruit and empty cells)
    (x, y) = coords
    side = 7    # square side
    shift = side // 2
    free_cells = {"Right": side**2, "Left": side**2, "Down": side**2, "Up": side**2}
    # left top corner of each square
    square_corners = {"Right": [x-shift, y+1], "Left": [x-shift, y-side], "Down": [x+1, y-shift], "Up": [x-side, y-shift]}
    for direction in square_corners:
      corner = square_corners[direction]
      if corner[0] < 0: corner[0] = 0
      elif corner[0] >= self.board_size[0]: corner[0] = self.board_size[0]
      if corner[1] < 0: corner[1] = 0
      elif corner[1] >= self.board_size[1]: corner[1] = self.board_size[1]
      for i in range(side):
        if i + corner[0] >= self.board_size[0]:
          continue
        free_cells[direction] -= self.cells[corner[0]+i][corner[1]:corner[1]+side].count("S")
      free_cells[direction] /= side**2
    return free_cells

  def is_goal(self, state):
    return state.center == self.fruit_coords

  def is_terminal(self, state):
    return self.snake.check_crash(state.center) or not self.valid_coords(state.center)

  def possible_actions(self, state):
    if self.is_terminal(state):
      return None
    forbidden_action = self.directions[self.snake.backward_move()]
    actions_list = ["Right", "Left", "Down", "Up"]
    actions_list.remove(forbidden_action)
    return actions_list

  def next_state(self, state, action):
    if self.is_terminal(state):
      return state
    (x, y) = state.center
    x += self.actions[action][0]
    y += self.actions[action][1]
    return self.create_state((x, y))

  def reward(self, state, action):
    next_state = self.next_state(state, action)
    if self.is_terminal(next_state):
      return -1.0
    elif self.is_goal(next_state):
      return 0.5
    return state.action_value(action)

  def rand_fruit_coords(self):
    empty_cells = []
    for i in range(self.board_size[0]):
      for j in range(self.board_size[1]):
        if self.cells[i][j] == ".":
          empty_cells.append((i, j))
    new_coords = empty_cells[random.randint(0, len(empty_cells)-1)]
    self.cells[self.fruit_coords[0]][self.fruit_coords[1]] = "."
    self.cells[new_coords[1]][new_coords[0]] = "F"
    return new_coords

  def reset_mdp(self):
    self.cells = [["." for j in range(self.board_size[1])]
                       for i in range(self.board_size[0])]
    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    self.cells[head_coords[0]][head_coords[1]] = "S"
    self.cells[head_coords[0] - 1][head_coords[1]] = "S"
    self.fruit_coords = (0, 0)  # need to have some initial value to use 'rand_fruit_coords'
    self.fruit_coords = self.rand_fruit_coords()
    self.snake.reset(head_coords)
    return self.create_state(head_coords)

class RLearning:
  def __init__(self, mdp, train=True, gamma=0.9, alpha=1.0, epsilon=0.0, episodes=10):
    self.mdp = mdp
    self.train = train
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    self.episodes = episodes

    self.epsilon_delta = 1/episodes
    self.q_table = {}

  def q_value(self, state, action):
    if state not in self.q_table:
      self.q_table[state] = {}
      self.q_table[state][action] = self.mdp.reward(state, action)
    elif action not in self.q_table[state]:
      self.q_table[state][action] = self.mdp.reward(state, action)
    return self.q_table[state][action]

  def max_q_value(self, state):
    possible_actions = self.mdp.possible_actions(state)
    if possible_actions is None:
      return -1.0
    max = self.q_value(state, "Right")
    for action in self.mdp.actions:
      q_val = self.q_value(state, action)
      if q_val > max:
        max = q_val
    return max

  def best_action(self, state):
    possbile_actions = self.mdp.possible_actions(state)
    if possbile_actions is None:
      return None

    probability = random.randint(0, 100)
    if self.train and self.epsilon*100 > probability:
      actions = list(self.mdp.actions.keys())
      actions.remove(self.mdp.directions[self.mdp.snake.backward_move()])
      return random.choice(actions)

    max = self.max_q_value(state)
    best_actions = []
    for action in self.mdp.actions:
      q_val = self.q_value(state, action)
      if q_val == max:
        best_actions.append(action)
    return random.choice(best_actions)

  def step(self, state):
    action = self.best_action(state)
    next_state = self.mdp.next_state(state, action)

    next_q_max = self.max_q_value(next_state)
    reward = self.mdp.reward(state, action)
    q_val = self.q_value(state, action)
    q_val += self.alpha * (reward + self.gamma * next_q_max - q_val)
    self.q_table[state][action] = q_val

    return (next_state, action)

