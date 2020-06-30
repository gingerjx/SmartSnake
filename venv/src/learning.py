import random
from snake import Snake
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque

actions = {"Right": (0, 1), "Left": (0, -1), "Down": (1, 0), "Up": (-1, 0)}
directions = {(0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"}
map_index_to_action = {0: "Right", 1: "Left", 2: "Down", 3: "Up"}
map_action_to_index = {"Right": 0, "Left": 1, "Down": 2, "Up": 3}

class State:
  def __init__(self, head, direction, is_eaten, obstacles, fruit_dir, walls_dist, body_dens, snake_len, head_tail_dist):
    self.head = head                      # tuple contains head position
    self.direction = direction            # tuple indicate direction of snake
    self.is_eaten = is_eaten              # boolean contains information about if fruit is eaten
    self.obstacles = obstacles            # dictionary contains information about obstacles presence in every four sides
    self.fruit_dir = fruit_dir            # dictionary contains information about fruit presence in every four sides
    self.walls_dist = walls_dist          # dictionary contains information about distance to wall in every four sides
    self.body_dens = body_dens            # dictionary contains infromation about density of snake's body segments in four sides
    self.snake_len = snake_len            # snake's length
    self.head_tail_dist = head_tail_dist  # manhattan distance between snake's head and tail

  def get_direction_vector(self):
    """Return one hot vector with direction"""
    vector = [0, 0, 0, 0]
    action = directions[self.direction]
    vector[map_action_to_index[action]] = 1
    return vector

  def to_net_input(self):
    """Return transformed state for network input"""
    return np.array(self.get_direction_vector() + list(self.obstacles.values()) + list(self.fruit_dir.values())
                    + list(self.walls_dist.values()) + list(self.body_dens.values()) + [self.snake_len, self.head_tail_dist]).reshape(1, -1)

class MDP:
  def __init__(self, board_size):
    self.board_size = board_size
    self.area = board_size[0] * board_size[1]
    self.reset()

  def init_cells(self):
    """Return initialized cells with two-segment snake and snake's coords.
       In cells: '.' - empty, 'S' - snake, 'F' - fruit."""
    cells = np.array([["." for j in range(self.board_size[1])]
                           for i in range(self.board_size[0])])
    head_coords = (self.board_size[0] // 2, self.board_size[1] // 2)
    init_snake_coords = (head_coords, (head_coords[0], head_coords[1] + 1))
    cells[init_snake_coords[0][0]][init_snake_coords[0][1]] = "S"
    cells[init_snake_coords[1][0]][init_snake_coords[1][1]] = "S"
    return cells, init_snake_coords

  def reset(self):
    """Reset environment"""
    self.cells, init_snake_coords = self.init_cells()
    self.snake = Snake(init_snake_coords, self.board_size)
    self.fruit_coords = (0, 0)  # need to init for 'rand_fruit_coords()' function
    self.fruit_coords = self.rand_fruit_coords()

  def is_out_of_board(self, coords):
    x, y = coords
    return x < 0 or x >= self.board_size[0] or y < 0 or y >= self.board_size[1]

  def rand_fruit_coords(self):
    """Return new position of fruit and update cells"""
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

  def get_state(self, coords, action):
    """Return state based on given 'coords', 'action' and current cells status"""
    x, y = coords

    obstacles = {}
    for act in actions:
      (dir_x, dir_y) = actions[act]
      next_coords = (x + dir_x, y + dir_y)
      if self.is_out_of_board(next_coords) or self.cells[next_coords[0]][next_coords[1]] == "S":
        obstacles[act] = 1.0
      else:
        obstacles[act] = 0.0

    fruit = {}
    fruit_x, fruit_y = self.fruit_coords
    fruit["Right"] = 1.0 if y < fruit_y else 0.0
    fruit["Left"] = 1.0 if y > fruit_y else 0.0
    fruit["Down"] = 1.0 if x < fruit_x else 0.0
    fruit["Up"] = 1.0 if x > fruit_x else 0.0

    is_eaten = (fruit_x == x and fruit_y == y)
    wall_distance = self.get_wall_distance(coords)
    body_dens = self.get_body_density(coords)
    snake_len = self.snake.eaten_fruits / self.area
    head_tail_dist = self.get_manhattan_distance(coords, self.snake.get_tail()) / self.area
    return State(coords, actions[action], is_eaten, obstacles, fruit, wall_distance, body_dens, snake_len, head_tail_dist)

  def get_wall_distance(self, coords):
    """Return distance to every four walls from 'coords'"""
    x, y = coords
    size_x, size_y = self.board_size
    distance = {}
    distance["Left"] = y/size_y
    distance["Right"] = (size_y - y - 1)/size_y
    distance["Up"] = x/size_x
    distance["Down"] = (size_x - x - 1)/size_x
    return distance

  def get_body_density(self, coords):
    """Return density of snake's body (body segments / view area) in each direction"""
    x, y = coords
    size = 5
    sft = size // 2
    shift = {"Right": (-sft, 1), "Left": (-sft, -size), "Down": (1, -sft), "Up": (-size, -sft)}
    body_density = {}
    for act in actions:
      x_sft, y_sft = shift[act]
      x_start = x + x_sft if x + x_sft > 0 else 0
      y_start = y + y_sft if y + y_sft > 0 else 0
      x_end = x_start + size
      y_end = y_start + size
      part = self.cells[x_start:x_end, y_start:y_end]
      body_density[act] = len(part[part == 'S']) / (size ** 2)
    return body_density

  def get_manhattan_distance(self, coords_1, coords_2):
    return abs(coords_1[0] - coords_2[0]) + abs(coords_2[1] - coords_2[1])

  def is_goal(self, state):
    return state.head == self.fruit_coords

  def is_terminal(self, state):
    return self.snake.is_crushed(state.head) or self.is_out_of_board(state.head)

  def get_opposite_action(self, state):
    return directions[(state.direction[0] * -1, state.direction[1] * -1)]

  def get_possible_actions(self, state):
    """Return possible actions in 'state'. If it's terminal return None"""
    if self.is_terminal(state):
      return None
    actions_list = list(actions.keys())
    actions_list.remove(self.get_opposite_action(state))
    return actions_list

  def get_next_state(self, state, action):
    """Return next state, which is result of taking 'action' in 'state'.
       If it's terminal return the same 'state'"""
    if self.is_terminal(state):
      return state
    (x, y) = state.head
    x += actions[action][0]
    y += actions[action][1]
    return self.get_state((x, y), action)

  def get_reward(self, state, action):
    """Return reward for taking 'action' in 'state'"""
    next_state = self.get_next_state(state, action)
    if self.is_terminal(next_state):
      return -1.0
    elif self.is_goal(next_state):
      return 0.5
    else:
      return 0.0

  def move_snake(self):
    """Based on given snake's direction snake and cells are updated"""
    head = self.snake.move()
    if head == self.fruit_coords:
      self.rand_fruit_coords()
      self.snake.eaten_fruits += 1
    else:
      tail = self.snake.pop_tail()
      self.cells[tail[0]][tail[1]] = "."
    self.cells[head[0]][head[1]] = "S"

class Agent:
  def __init__(self, mdp, train, episodes, gamma, alpha, epsilon, max_steps):
    self.mdp = mdp
    self.current_state = self.mdp.get_state(self.mdp.snake.get_head(), "Left")
    self.train = train
    self.gamma = gamma            # discount factor
    self.alpha = alpha            # learning rate
    self.epsilon = epsilon if train else 0       # epsilon-greedy action probability
    self.episodes = episodes if train else 1     # number of episodes
    self.max_steps = max_steps    # max useless (without eating fruit) steps in episode

    self.epsilon_decay = 0.9
    self.epsilon_min = 0.01
    self.memory = deque(maxlen=2000)  # memory of our agent - s, a, r, s', t||e
    self.training_period = 20         # the number of steps followed by training
    self.training_counter = 0         # counter of taken steps
    self.target_update_period = 10    # the number of online network training to copy weights into online
    self.target_update_counter = 0    # counter of online network training
    self.batch_size = 64

    self.create_model()

  def create_model(self):
    "If its training new model is creted, otherwise it is loaded from file"
    if self.train:
      """Online network"""
      self.online = Sequential()
      self.online.add(Dense(16, input_dim=self.current_state.to_net_input().shape[1], activation='relu'))
      self.online.add(Dense(32, activation='relu'))
      self.online.add(Dense(4, activation='linear'))
      self.online.compile(loss='mse', optimizer=Adam(lr=self.alpha))
      """Target network, same weights and structure as in online.
         This network is used only in get_max_target_predictions()"""
      self.target = clone_model(self.online)
      self.target.set_weights(self.online.get_weights())
    else:
      self.online = load_model('network_model.h5', compile=False)
      self.online.compile(loss='mse', optimizer=Adam(lr=self.alpha))
      self.target = load_model('network_model.h5', compile=False)

  def save_model(self):
    self.target.save('network_model.h5')

  def get_action(self, state):
    """Return None if there's no possible action.
       If it's training it will return epsilon-greedy action or the best one from net.
       If it's normal game it will return the best action from net"""
    possbile_actions = self.mdp.get_possible_actions(state)
    if possbile_actions is None:
      return None

    probability = random.uniform(0, 1)
    if self.train and self.epsilon > probability:
      return random.choice(possbile_actions)

    net_action = self.get_net_action(state, possbile_actions)
    if net_action is None:
      return random.choice(possbile_actions)
    else:
      return net_action

  def get_net_action(self, state, possible_actions):
    """From network output for 'state' it finds the best actions.
       From intersection of net best actions and 'possbile_actions' we random one action to be returned.
       If there isn't common actions, None is returned"""
    prediction = self.target.predict(state.to_net_input())
    best_actions_index = np.where(prediction == np.max(prediction.reshape(-1)))[1]
    best_actions = [map_index_to_action[b_a_i] for b_a_i in best_actions_index]
    intersection = [v for v in best_actions if v in possible_actions]
    if not intersection:  # empty
      return None
    else:
      return random.choice(intersection)

  def net_training(self):
    """Train network based on samples from memory"""
    if self.batch_size > len(self.memory):
      return
    self.target_update_counter += 1
    self.decrease_epsilon()
    samples = random.sample(self.memory, self.batch_size)
    batch, action, next_batch, reward, is_term_or_eaten = self.create_batch(samples)

    max_next = self.get_max_target_predictions(next_batch)
    expected = self.get_expected_output(is_term_or_eaten, reward, max_next)
    predictions = self.online.predict(batch)

    """Prediction in 's' is updated at 'a' output by expected value.
       In case of forbidden action, prediction is assign to 0 there."""
    for i in range(self.batch_size):
      predictions[i][map_action_to_index[action[i]]] = expected[i]
      forbidden_action = self.mdp.get_opposite_action(samples[i][0])
      action_index = map_action_to_index[forbidden_action]
      predictions[i][action_index] = 0

    """Training of online network"""
    self.online.fit(batch, predictions, epochs=20, verbose=0)
    """Update of target network (copy weights from online)"""
    if self.target_update_counter >= self.target_update_period:
      self.target.set_weights(self.online.get_weights())
      self.target_update_counter = 0

  def create_batch(self, samples):
    """Return separated state, action, reward, next state and is_terminal or is_eaten from given 'samples'"""
    zipped_samples = list(zip(*samples))
    batch = [v.to_net_input() for v in zipped_samples[0]]
    action = zipped_samples[1]
    reward = zipped_samples[2]
    next_batch = [v.to_net_input() for v in zipped_samples[3]]
    is_term_or_goal = zipped_samples[4]
    return np.array(batch).reshape(self.batch_size, -1), np.array(action), \
           np.array(next_batch).reshape(self.batch_size, -1), np.array(reward), np.array(is_term_or_goal)

  def get_max_target_predictions(self, batch):
    """Returns values of best target net predictions"""
    predictions = self.target.predict(batch)
    max_values = np.max(predictions, axis=1)
    return max_values

  def get_expected_output(self, is_term_or_eaten, reward, max_values):
    """Return calculated expected value of specific action for network training"""
    expected_output = []
    for i in range(self.batch_size):
      if is_term_or_eaten[i]:  expected_output.append(reward[i])
      else:                    expected_output.append(reward[i] + self.gamma*max_values[i])
    return expected_output

  def step(self):
    """Based on get_action() choose next_state and update snake's direction.
       Append tuple (s, a, r, s', t||e) to memory. Update agent."""
    action = self.get_action(self.current_state)
    assert action is not None

    next_state = self.mdp.get_next_state(self.current_state, action)
    reward = self.mdp.get_reward(self.current_state, action)
    self.memory.append((self.current_state, action, reward, next_state,\
                        self.mdp.is_terminal(next_state) or next_state.is_eaten))  # is_terminal or is_eaten

    if self.training_counter >= self.training_period and self.train:
      self.training_counter = 0
      self.net_training()

    self.mdp.snake.direction = actions[action]
    self.current_state = next_state
    self.training_counter += 1

  def reset_episode(self):
    """Reset enviroment and agent for new episode"""
    self.mdp.reset()
    self.current_state = self.mdp.get_state(self.mdp.snake.get_head(), "Left")

  def is_terminal(self):
    return self.mdp.is_terminal(self.current_state)

  def is_goal(self):
    return self.mdp.is_goal(self.current_state)

  def get_score(self):
    """Return number of eaten fruits in episode"""
    return self.mdp.snake.eaten_fruits

  def decrease_epsilon(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    else: self.epsilon = self.epsilon_min