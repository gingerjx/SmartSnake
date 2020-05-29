import random
import snake

class State:
  def __init__(self, center, obstacles, reward, walls_dist, body_dist):
    self.center = center          # tuple contains head position
    self.obstacles = obstacles    # dictionary contains information about obstacles presence in every four sides
    self.reward = reward          # dictionary contains information about reward presence in every four sides
    self.walls_dist = walls_dist  # dictionary contains information about distance to wall in every four sides
    self.body_dist = body_dist    # dictionary contains information about distance to body in every four sides

  def equals(self, state):
    return self.center == state.center and\
           self.obstacles == state.obstacles and\
           self.reward == state.reward and \
           self.free_cells == state.free_cells
  # temporary function
  def action_value(self, action):
    obstacles_value = -1.0 if self.obstacles[action] else 0.0
    reward_value = 0.2 if self.reward[action] else 0.0
    walls_dist_value = self.walls_dist[action]/10
    body_dist_value = self.body_dist[action]/10
    return (obstacles_value + reward_value + walls_dist_value + body_dist_value)/4

class Mdp:
  def __init__(self, board_size):
    self.board_size = board_size
    self.cells = [["." for j in range(self.board_size[1])]
                       for i in range(self.board_size[0])]
    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    init_coords = (head_coords, (head_coords[0], head_coords[1] + 1))
    self.cells[init_coords[0][0]][init_coords[0][1]] = "S"
    self.cells[init_coords[1][0]][init_coords[1][1]] = "S"
    self.snake = snake.Snake(init_coords, board_size)
    self.fruit_coords = (0, 0)  # need to init for 'rand_fruit_coords()' function
    self.fruit_coords = self.rand_fruit_coords()
    self.actions = {"Right": (0, 1), "Left": (0, -1), "Down": (1, 0), "Up": (-1, 0)}
    self.directions = {(0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"}

  def out_of_board(self, coords):
    (x, y) = coords
    return x < 0 or x >= self.board_size[0] or y < 0 or y >= self.board_size[1]

  def rand_fruit_coords(self):
    empty_cells = []
    for i in range(self.board_size[0]):
      for j in range(self.board_size[1]):
        if self.cells[i][j] == ".":
          empty_cells.append((i, j))
    new_coords = random.choice(empty_cells)
    self.cells[self.fruit_coords[0]][self.fruit_coords[1]] = "."
    self.cells[new_coords[0]][new_coords[1]] = "F"
    self.fruit_coords = new_coords
    return new_coords

  def create_state(self, coords):
    (x, y) = coords

    obstacles = {}
    for action in self.actions:
      (dir_x, dir_y) = self.actions[action]
      next_coords = (x + dir_x, y + dir_y)
      if self.out_of_board(next_coords) or self.cells[next_coords[0]][next_coords[1]] == "S":
        obstacles[action] = True
      else:
        obstacles[action] = False

    reward = {}
    (fruit_x, fruit_y) = self.fruit_coords
    reward["Right"] = True if y < fruit_y else False
    reward["Left"] = True if y > fruit_y else False
    reward["Down"] = True if x < fruit_x else False
    reward["Up"] = True if x > fruit_x else False

    wall_distance = self.wall_distance(coords)
    body_distance = self.body_distance(coords)
    return State(coords, obstacles, reward, wall_distance, body_distance)

  def wall_distance(self, coords):
    (x, y) = coords
    (size, _) = self.board_size
    distance = {}
    distance["Left"] = y/size
    distance["Right"] = (size - y - 1)/size
    distance["Up"] = x/size
    distance["Down"] = (size - x - 1)/size
    return distance

  def body_distance(self, coords):
    (center_x, center_y) = coords
    (size, _) = self.board_size
    distance = {}
    for action in self.actions:
      (dir_x, dir_y) = self.actions[action]
      x, y = center_x, center_y
      distance[action] = 0
      while True:
        x += dir_x
        y += dir_y
        if self.out_of_board((x, y)):
          break
        if self.snake.crush((x, y)):
          break
        distance[action] += 1
      distance[action] /= size

    return distance

  def is_goal(self, state):
    return state.center == self.fruit_coords

  def is_terminal(self, state):
    return self.snake.crush(state.center) or self.out_of_board(state.center)

  def possible_actions(self, state):
    if self.is_terminal(state):
      return None
    actions_list = list(self.actions.keys())
    forbidden_action = self.directions[self.snake.backward_move()]
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

  def reset(self):
    self.cells = [["." for j in range(self.board_size[1])]
                  for i in range(self.board_size[0])]
    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    init_coords = (head_coords, (head_coords[0], head_coords[1] + 1))
    self.cells[init_coords[0][0]][init_coords[0][1]] = "S"
    self.cells[init_coords[1][0]][init_coords[1][1]] = "S"
    self.snake.reset(init_coords)
    self.fruit_coords = (0, 0)  # need to init for 'rand_fruit_coords()' function
    self.fruit_coords = self.rand_fruit_coords()

class RLearning:
  def __init__(self, mdp, train=True, gamma=0.9, alpha=1.0, epsilon=0.8, episodes=25, max_steps=100):
    self.mdp = mdp
    self.state = self.mdp.create_state(self.mdp.snake.body[0])
    self.train = train
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    self.episodes = episodes
    self.max_steps = max_steps

    self.epsilon_delta = self.epsilon/episodes
    self.q_table = [{}, {}]    # two tables for Double QLearning

  def q_value(self, state, action, tab_num):
    # return value from q_table for table 0 or 1 ('tab_num'), if it doesn't exist assign reward to it
    if state not in self.q_table[tab_num]:
      self.q_table[tab_num][state] = {}
      self.q_table[tab_num][state][action] = self.mdp.reward(state, action)
    elif action not in self.q_table[tab_num][state]:
      self.q_table[tab_num][state][action] = self.mdp.reward(state, action)
    return self.q_table[tab_num][state][action]
  def double_q_value(self, state, action):
    # return sum of values from q_table 0 and 1
    return self.q_value(state, action, 0) + self.q_value(state, action, 1)

  def max_q_value(self, state, tab_num):
    # return max value from table 0 or 1 ('tab_num')
    possible_actions = self.mdp.possible_actions(state)
    if possible_actions is None:
      return -1.0
    max = self.q_value(state, "Right", tab_num)
    for action in self.mdp.actions:
      q_val = self.q_value(state, action, tab_num)
      max = q_val if q_val > max else max
    return max
  def double_max_q_value(self, state):
    # return max value from sum of tables 0 and 1
    possible_actions = self.mdp.possible_actions(state)
    if possible_actions is None:
      return -1.0
    max = self.double_q_value(state, "Right")
    for action in self.mdp.actions:
      q_val = self.double_q_value(state, action)
      max = q_val if q_val > max else max
    return max

  def argmax_q_value(self, state, tab_num):
    # return max action from table 0 or 1 ('tab_num')
    max = self.max_q_value(state, tab_num)
    best_actions = []
    for action in self.mdp.actions:
      if self.q_value(state, action, tab_num) == max:
        best_actions.append(action)
    return random.choice(best_actions)
  def double_argmax_q_value(self, state):
    # return max action from sum of tables 0 and 1
    max = self.double_max_q_value(state)
    best_actions = []
    for action in self.mdp.actions:
      if self.double_q_value(state, action) == max:
        best_actions.append(action)
    return random.choice(best_actions)

  def best_action(self, state):
    # if state is terminal return None
    # sometimes return random action (depends on epsilon)
    # otherwise return action based on max value from both tables
    possbile_actions = self.mdp.possible_actions(state)
    if possbile_actions is None:
      return None

    probability = random.randint(0, 100)
    if self.train and self.epsilon*100 > probability:
      actions = list(self.mdp.actions.keys())
      actions.remove(self.mdp.directions[self.mdp.snake.backward_move()])
      return random.choice(actions)

    return self.double_argmax_q_value(state)

  def step(self):
    # Choose best action, update q_table 0 or 1, return new state and taken action
    action = self.best_action(self.state)
    next_state = self.mdp.next_state(self.state, action)

    self.update(next_state, action)

    self.state = next_state
    self.mdp.snake.direction = self.mdp.actions[action]
    return (self.state, action)

  def update(self, next_state, action):
    reward = self.mdp.reward(self.state, action)
    tab_num = 1 if random.randint(0, 100) > 50 else 0  # 50% for 0 and 50% for q_table 1
    max_action = self.argmax_q_value(next_state, tab_num)
    next_q_val = self.q_value(next_state, max_action, 1-tab_num)
    q_val = self.q_value(self.state, action, tab_num)
    q_val += self.alpha * (reward + self.gamma * next_q_val - q_val)
    self.q_table[tab_num][self.state][action] = q_val