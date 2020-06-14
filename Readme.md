# Reinforcement Learning

Performance of Reinforcement Learning in snake game

## Language and libraries

Python 3.6, pygame 1.9.6, Tensorflow 2.2.0, Keras 2.3.1

## Game view

<img src='/snakeEpoc.gif'/>

## About rl

In the future...

## Done

For now snake game is ready with all GUI components. Snake's moves are controlled by Double Q Neural Network. During training, moves are chosen from network prediction or by epsilon-greedy approach. State is represented as Function Approximation features and they are transformed to vector shape, prepared for networks' input. In every step snake saves in memory - current state, taken action, received reward, next state and boolean saying if it's terminal or goal state. After some steps online network is trained by using samples from the memory. There is two networks, online and target, nothing changed in relation of before implementation (approach with one network, online network here) except that maxQ(s',a') is taken from target network and it is updated after few training of online network by copying weights from online to target.
It looks a little better, but it isn't final result.
