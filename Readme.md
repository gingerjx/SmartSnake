# Reinforcement Learning

Performance of Reinforcement Learning in snake game

## Language and libraries

Python 3.6, pygame 1.9.6

## Game view

<img src='/snakeEpoc.gif'/>

## About rl

In the future...

## Done & TODO

**Done**

For now snake game is ready with all GUI components. Snake's moves are controlled by Deep Q Learning. During training, moves are chosen from network prediction or by epsilon-greedy approach. State is represented as Function Approximation features and they are transformed to vector shape, prepared for networks' input. In every step snake saves in memory - current state, taken action, received reward, next state and boolean saying if it's terminal or goal state. After some steps network is trained by using samples from the memory.
For now this implementation works as well as Function Approximation, but future changes should improve this snake and makes him smarter!

**TODO**

Try another state representation
Implement Double Deep Q Learning
