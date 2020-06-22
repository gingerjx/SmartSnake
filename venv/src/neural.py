import random
import numpy as np

"""Activation functions"""
def emptyFunc(x):
  return x

def ReLU(x):
  return x * (x > 0)

def derivReLU(x):
    return np.ones(x.shape) * (x>0)

"""Neural Network class"""
class Network:
  def __init__(self):
    self.layers = np.array([], dtype=int)       # list contains number of neurons in each layer
    self.activations = np.array([emptyFunc])    # list contatins activation function of each layer
    self.derivatives = np.array([emptyFunc])    # list contatins derivatives of each layer
    self.weights = []
    Network.functions = {'empty': (emptyFunc, emptyFunc), 'relu': (ReLU, derivReLU)}
    Network.reverse_functions = {emptyFunc: 'empty', ReLU: 'relu'}

  def copy(self):
    """Return copied network"""
    net_copy = Network()
    net_copy.layers = self.layers.copy()
    net_copy.set_weights(self.weights.copy())
    net_copy.activations = self.activations.copy()
    net_copy.derivatives = self.derivatives.copy()
    return net_copy

  def set_weights(self, weights):
    self.weights = weights

  def get_weights(self):
    return self.weights.copy()

  def add_input(self, number):
    """Add input layer to network structure"""
    self.layers = np.append(self.layers, number)

  def add_fully_connected(self, neurons, function='empty', left_range=-0.1, right_range=0.1):
    """Add fully connected layer to network structure"""
    self.layers = np.append(self.layers, neurons)
    self.activations = np.append(self.activations, Network.functions[function][0])
    self.derivatives = np.append(self.derivatives, Network.functions[function][1])
    weights2D = []
    for j in range(neurons):
      weights1D = []
      for k in range(self.layers[-2]):
        weight_value = np.random.uniform(left_range, right_range)
        weights1D.append(weight_value)
      weights2D.append(weights1D)
    self.weights.append(np.array(weights2D))

  def predict(self, input_data):
    """Return prediction of neural network based on given 'input_data'"""
    output = input_data
    for i in range(len(self.weights)):
      output = output @ np.transpose(self.weights[i])
      output = self.activations[i](output)
    return output

  def fit(self, input_data, expected_data, iterations=100, alpha=0.01, batch_size=-1, drop_percent=0.5):
    """Train network based on given arguments"""
    assert input_data.shape[0] == expected_data.shape[0]
    assert batch_size <= input_data.shape[0]
    if batch_size == -1:
      batch_size = input_data.shape[0]
    """Start iterating"""
    for it in range(iterations):
      for bIt in range( int(input_data.shape[0]/batch_size) ):
        """Create batch"""
        batch_input = []
        batch_expected = []
        if (bIt+1)*batch_size > input_data.shape[0]:
          batch_input = input_data[bIt * batch_size:]
          batch_expected = expected_data[bIt * batch_size:]
        else:
          batch_input = input_data[bIt*batch_size:(bIt+1)*batch_size]
          batch_expected = expected_data[bIt * batch_size:(bIt + 1) * batch_size]
        """Create dropout mask"""
        dropout_mask = np.zeros((batch_input.shape[0], self.layers[1]))
        for vector in dropout_mask:
          for i in range(int(len(vector) * drop_percent)):
            vector[i] = 1.0
          np.random.shuffle(vector)
        ratio = 1.0 / drop_percent
        scaled_dropout_mask = dropout_mask * ratio
        """Calculate layer outputs"""
        layer_outputs = [batch_input]
        for layIt in range(1, len(self.layers)):
          layer_outputs.append(layer_outputs[layIt - 1] @ np.transpose(self.weights[layIt - 1]))
          layer_outputs[layIt] = self.activations[layIt - 1](layer_outputs[layIt])
          if (layIt == 1 and len(self.layers) > 2):
            layer_outputs[layIt] *= scaled_dropout_mask
        """Calculate delta"""
        layer_delta = [0 for i in range(len(self.layers))]
        layer_delta[-1] = layer_outputs[-1] - batch_expected
        layer_delta[-1] /= batch_size
        for layIt in range(len(self.layers) - 2, 0, -1):
          layer_delta[layIt] = layer_delta[layIt + 1] @ self.weights[layIt]
          layer_delta[layIt] *= self.derivatives[layIt - 1](layer_outputs[layIt])
          if layIt != 0 and layIt != len(self.layers) - 1:
            layer_delta[layIt] *= dropout_mask
        """Calculate weighted delta"""
        layer_weighted_delta = [0 for i in range(len(self.layers))]
        for layIt in range(len(self.layers) - 1, 0, -1):
          layer_weighted_delta[layIt] = np.transpose(layer_delta[layIt]) @ layer_outputs[layIt - 1]
        """Update weights"""
        for layIt in range(len(self.layers) - 1, 0, -1):
          self.weights[layIt - 1] -= layer_weighted_delta[layIt] * alpha

  def save(self):
    """Save network in files - ..structure and ..weights[num] where num is number of layer"""
    with open('../saved_net/trained_net_structure', 'w') as file:
      """Structure save format:
          number of layers
          number of neurons in each layer
          activation function in each layer"""
      file.write(str(len(self.layers)) + '\n')
      for lay in self.layers:
        file.write(str(lay) + ' ')
      file.write("\n")
      for activ in self.activations:
        file.write(Network.reverse_functions[activ] + ' ')
    for i in range(len(self.layers)-1):
      """Save weights 2D arrays in .csv, each layer has it's own file"""
      np.savetxt('../saved_net/trained_net_weights' + str(i) + '.csv', self.weights[i], delimiter=',')

def load():
  """Load network from files created by 'save()' method in Network class"""
  network = Network()
  with open('../saved_net/trained_net_structure', 'r') as file:
    layers_num = int(file.readline())
    layers = [int(x) for x in file.readline().split()]
    functions = [x for x in file.readline().split()]
    network.layers = np.array(layers)
    for func in functions:
      network.activations = np.append(network.activations, Network.functions[func][0])
      network.derivatives = np.append(network.derivatives, Network.functions[func][1])
  for i in range(layers_num-1):
    weights = np.loadtxt('../saved_net/trained_net_weights' + str(i) + '.csv', delimiter=',')
    network.weights.append(weights)
  return network