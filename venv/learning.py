import random
import snake

class State:
  def __init__(self, center, obstacles, reward, goal, walls_dist):
    self.center = center          # tuple contains head position
    self.obstacles = obstacles    # dictionary contains information about obstacles presence in every four sides
    self.reward = reward          # dictionary contains information about reward presence in every four sides
    self.goal = goal              # boolean contatins information about if state is a goal state
    self.walls_dist = walls_dist  # dictionary contains information about distance to wall in every four sides

  def features_value(self, action):
    obstacles_value = 1.0 if self.obstacles[action] else 0.0
    reward_value = 0.1 if self.reward[action] else 0.0
    goal_value = 1.0 if self.goal else 0.0
    walls_dist_value = self.walls_dist[action]
    return {0: obstacles_value, 1: reward_value, 2: goal_value,
            3: walls_dist_value}

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

    goal = fruit_x == x and fruit_y == y

    wall_distance = self.wall_distance(coords)
    return State(coords, obstacles, reward, goal, wall_distance)

  def wall_distance(self, coords):
    (x, y) = coords
    (size, _) = self.board_size
    distance = {}
    distance["Left"] = y/size
    distance["Right"] = (size - y - 1)/size
    distance["Up"] = x/size
    distance["Down"] = (size - x - 1)/size
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
    else:
      return 0.0

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
  def __init__(self, mdp, train=True, gamma=0.9, alpha=1.0, epsilon=0.8, episodes=25, max_steps=100, l_range=-0.1, r_range=0.1):
    self.mdp = mdp
    self.state = self.mdp.create_state(self.mdp.snake.body[0])
    self.train = train
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    self.episodes = episodes
    self.max_steps = max_steps

    self.l_range = l_range
    self.r_range = r_range
    self.epsilon_delta = self.epsilon/episodes
    self.wei_num = 4
    self.weights = [l_range + (random.random() * (r_range - l_range)) for i in range(self.wei_num)]

  def reset_rl(self, episode):
    self.epsilon += self.epsilon_delta * episode
    self.weights = [self.l_range + (random.random() * (self.r_range - self.l_range)) for i in range(self.wei_num)]

  def function_approximation(self, state, action):
    features = state.features_value(action)
    q_value = 0
    for i in range(len(features)):
      q_value += features[i]*self.weights[i]
    return q_value

  def max_q_value(self, state):
    possible_actions = self.mdp.possible_actions(state)
    if possible_actions is None:
      return -1.0
    max = self.function_approximation(state, "Right")
    for action in self.mdp.actions:
      q_val = self.function_approximation(state, action)
      max = q_val if q_val > max else max
    return max

  def argmax_q_value(self, state):
    max = self.max_q_value(state)
    best_actions = []
    for action in self.mdp.actions:
      if self.function_approximation(state, action) == max:
        best_actions.append(action)
    return random.choice(best_actions)

  def best_action(self, state):
    possbile_actions = self.mdp.possible_actions(state)
    if possbile_actions is None:
      return None

    probability = random.randint(0, 100)
    if self.train and self.epsilon*100 > probability:
      actions = list(self.mdp.actions.keys())
      actions.remove(self.mdp.directions[self.mdp.snake.backward_move()])
      return random.choice(actions)

    return self.argmax_q_value(state)

  def temporal_difference(self, next_state, action):
    next_q_max = self.max_q_value(next_state)
    reward = self.mdp.reward(self.state, action)
    q_val = self.function_approximation(self.state, action)
    return reward + self.gamma * next_q_max - q_val

  def update(self, next_state, action):
    delta = self.temporal_difference(next_state, action)
    features = self.state.features_value(action)
    for i in range(len(features)):
      self.weights[i] += self.alpha * delta * features[i]

  def step(self):
    action = self.best_action(self.state)
    next_state = self.mdp.next_state(self.state, action)
    self.update(next_state, action)
    self.state = next_state
    self.mdp.snake.direction = self.mdp.actions[action]
    return (self.state, action)