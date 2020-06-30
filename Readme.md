# Reinforcement Learning

Performance of Reinforcement Learning in snake game

## Language and libraries

Python 3.6 <br/>
pygame 1.9.6 <br/>
numpy 1.19.0 <br/>
tensorflow 2.2.0 <br/>
Keras 2.4.3 <br/>

## Game view

<img src='/img/animation.gif'/>

## About project

### Few terms
It is project about smart snake, who doesn't need to human control. <br/>
Snake is led by Double Deep Q Learning Agent. What does it mean? <br/>
Let's say mentioned <b>agent</b> is a head of our snake, brain of our pet. <br/>
His task is to eat as much as possible fruits and try not to crush. This task in reinforcement learning is called <b>goal</b>. <br/>
Everything what snake is able to get know is an <b>environment</b> in this approach terms. In other words, it is just a board and everything what's on it. <br/>
Snake can moves in four directions (right, left, down and up). This moves is agent <b>actions</b>. <br/>
In every step snake takes an action and gets <b>reward</b> for that. For example if move (action) provides him to fruit he will get some positive reward. In situation when action provides to crush we will punish him by giving some negative reward. It is like learning to fetting a dog. If he brings back a stick, he receive a snack, otherwise he can just imagine how it tastes. <br/>
Another core term in reinforcement learning is <b>state</b>. State is some kind of view what is going on in environment at the moment. In the simplest way state can be represents by frame from game. It is saying us what's exactly  happens in this time step. Knowing state, snake can decide what he should do - what action to take. <br/>
When snake is crushed on his tail or on wall, this state's called <b>terminal</b>. That one game's called <b>episode</b> <br/>
And the last term is <b>training</b>. Training consists of certain number of episodes, in which snake's learning how to play.

### State representation
Since we know what is state, i will show you how I defined it. At the previous chapter I said the simplest way to implement state is just a frame game. It is common approach and in some cases effective also, but I decided to use game features instead of passing whole board. <br/> Features: <br/>
- Snake's move direction.
- Information about presence of any obstacle next to him in every direction.
- Information about presence of fruit in every direction.
- Information about distance to wall in every direction.
- Information about amount of snake's body segments in every direction.
- Snake's length.
- Distance between head and tail

### Q Learning
We completely skipped Double Deep Q Learning term. Let's start from the end. <br>
Let's imagine we are waking up in totally dark room. We see nothing so we're going in some random directions, to <b>explore</b> this room. Firstly we will hit the shelf by head, then we will step on the Lego brick. Nothing nice. But at the end we will find switch, and after turning on the light structure of room is no longer a threat. The second scenario is that you will fall down the stairs and break your neck. <br/>
Exactly in same situation is our snake. Initially he wakes up on board and know nothing. He will take some random action and move through the board. Consequence of each step is a reward (hitting the shelf in example above). And in the end he will reach terminal state (reaching switch or falling down the stairs) and finish episode. But in every step he will keep in mind how good was taking specific action in specific state. In next trip in dark room person will remember that in some places he should avoid shelves or Lego bricks. And the most important thing how to survive without breaking neck on stairs. <br/><br/>

And here we comes with <b>Q Learning</b>. This approach provides us information about quality of move in specific state. In code we can represents some number value on state-action pair in container - <b>Q Table</b>. But as we said, at the beginning snake have no information about any state except that initial one. That's why at start he will explore environment and get know some states by moving randomly. And at every step he will assign value to state-action pair. But what value should he assign? Here i will show Bellman Equation

<img src='/img/bellman.jpg'/>

Looks horrible, but only looks. We are taking action <b>a</b> in state <b>s</b> and trying to estimate <b>Q(s,a)</b> value. We will use next state <b>s'</b>, which is result of taking <b>a</b> in <b>s</b>. In this next state, taking into account all possible actions in <b>s'</b>, we are looking for max value of <b>Q(s',a')</b>. In simple words, we are trying to estimate, what good is waiting for us there. Gamma (between 0 and 1) - <b>discount factor</b> - says us how much should we take into account this calculated value. Next we are adding reward value for taking action in state and subtract current <b>Q(s,a)</b>. Result of this equation is multiplied by alpha (between 0 and 1) - <b>learning rate</b>. It helps us to control pace of learning. After all we are adding calculated value to current <b>Q(s,a)</b> and that's how we get new, updated <b>Q(s,a)</b>. If we haven't value of <b>Q(s,a)</b> or <b>Q(s',a')</b> in our <b>Q Table</b>, we assign it to 0 and then continue our calculations. <br/><br/>

After we explore environment a little, we can start to <b>exploit</b> it. Exploitation means, look at Q Table and choose action in current state, which have the highest value there. We will fully exploit during normal game, but during training after initial stage, we need to exploit and explore alternately. That's why we using <b>epsilon greedy</b>. Epsilon is a parameter between 0 and 1, which says about probability of taking random action instead of using Q Table. If we would only exploit, we would learn nothing. Epsilon greedy approach provides us some unexpected behavior, which leads to learn new ways to reach a goal. For example there is more than one way to reach the switch or fruit in the snake game.

### Deep Q Learning

Unfortunately Q Learning is not sufficient for large environments. Imagine having environment with 1 milion of states and 1000 actions in each state. Our Q Table would be really large, too large. But not only memory is a problem, training of this model would take very very long time and it is the main issue. Neural networks can resolve of our problem. From now Q Table has been replaced by network. Everything looks similar, we are just passing state to our network and in the output we have 4 values corresponding to 4 actions in game. For training purposes we are adding also memory which will store information about last 2000 steps. Information:
- current state
- taken action
- received reward
- next state
- boolean saying if it's terminal state or state where fruit is eaten

During training, we are taking few samples from memory. Network will be trained on this samples. Choosing samples instead of whole memory will speed up our training cause of less information to process and
break correlation. <b>Temporal difference</b> - reward + maxQ(s',a') - is a part from Bellman equation, where maxQ(s',a') is the highest value of next state net's output. Having value of temporal difference we're taking and modifying output of current state in argmaxQ(s',a') by temporal difference. Last thing to do with output of current state is mark forbidden action (turning back in snake game) as 0.
Now we can pass this modified output as an expected and train our network.

### Double Deep Q Learning

We haven't said that, but so far we were using <b>online network</b>. It means that every operation like prediction or training was made on one network. It can leads to discrepancy of our network, which is undesirable. That's why we are adding second network - <b>target network</b>. Initially online and target are the same. But during training only online is modified. After some amount of trainings, we are copying weights from online to target network. The last modification is temporal difference, which we said about above. Instead of using online network to find the highest value from the next state, we are using target network here.

### Summary

It is briefly description mechanisms used to learn our snake how to be smart. It is also possible to improve this project and apply another/more techniques like for example convolutional version of Double Deep Q Learning. I decided to stop at this point, cause i think this approach is sufficient for this simple game.
