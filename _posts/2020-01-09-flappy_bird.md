---
title: "Deep Learning - Flappy Bird"
date: 2020-01-09
tags: [deep learning, games, computer vision, artificial intelligence]
header:
  image: "/images/data_art.png"
excerpt: "A Deep Learning bot to play Flappy Bird game"
classes: wide
---

<style>
i {
    color: #f25278;
}

b {
    color: #f25278;
}

body {
    text-align: justify;
    font-size: 18px;
}
</style>

## A Deep learning bot to play Flappy Bird game.

<!-- todo: add demo videos and images and snippets of DQN and Brain classes. -->

<b>Tools Used</b>
<ul>
    <li>Python 3.6</li>
    <li>Tensorflow 1.12.3</li>
    <li>Keras 2.2.4</li>
    <li>Pygame 1.9.6</li>
</ul>


Flappy Bird game released in 2013 for iOS and Android instantly gained popularity worldwide due to its difficulty to score. 
The goal of the game is to reach as far as possible without hitting the obstacles. The player controlled bird moves right with constant velocity and is pulled down by gravity. 
The player must tap the screen to make the bird flap its wings and go 
higher to avoid the obstacles (pipes).

The game environment used in this project is highly influenced on this 
existing implementation of Flappy Bird made using <i>pygame</i> in 
<i>Python</i> and available open source on this <a href="https://github.com/sourabhv/FlapPyBird.git">Github repository</a>.

<b>Deep Q Network</b>

On our <i>first</i> attempt, we used a Deep Q Network as the agent to learn and play the game. 
The Deep Q Network is a type of Reinforcement Learning algorithm. 
In Reinforcement Learning, an agent interacts with an environment arriving at different 
scenarios known as states by performing actions influenced by reward for each action 
providing each state follows the <a href="https://en.wikipedia.org/wiki/Markov_model"><i>Markov model</i></a>.
The basic principle used in Q Learning is that at any given state, an action is performed that will eventually yield the highest cumulative reward.
This Q-value is calculated using Bellman equation given as 

<i>Q(s, a) = r(s, a) + γ * max Q(s',a)</i> 

where <i>s</i> is the current state, 

<i>a</i> is the performed action, 

<i>r(s,a)</i> is the immediate reward received for that state <i>s</i> and action <i>a</i>, 

<i>s'</i> is the next state achieved after taking action <i>a</i>, 

<i>max Q(s', a)</i> represents the maximum Q-value for next state <i>s'</i>,and

<i>γ</i> is the discount factor used to control influence of future reward.

For the Deep Q Network, we used a Convolutional Neural Network (CNN) model which takes input as a stack of 
4, 80 X 80 px consecutive frames. The 4 frames were used to let the model to learn the downward movement of bird.
The CNN model used contained 3 convolutional layers, 1 hidden layer and 1 output layer. 
The rectified linear unit (ReLU) activation function is used for the first 4 layers and weighted output for the last layer.
To train the model, a memory which can hold data of upto 50000 frames was used. 
To collect enough data to train the model, the agent was run for 50000 steps 
without training with a random action every 1 out of 100 steps.<!-- Check this with epsilon value-->
The model was trained every 10000 steps with a batch size of 32 for a total more than 1 million steps. 
Each cell in memory contained the current frame as the image, action used, reward obtained, next frame resulting from the action and 
However, the agent was still not able to pass more than 15 pipes consecutively. 
We also tried a range of values for hyperparameters but the model failed to converge. 
In short, the score obtained was not satisfactory. Hence, we discarded this model.

---

<b>Artifical Neural network</b>

On our <i>second</i> attempt, we used a Deep Learning Regression model as the agent to play the game.
This model used the bird's position in y-direction, the gap between bird and the top obstacle in y-direction, 
the distance between bird and upcoming pipe in x-direction and the downward velocity of bird in y-direction.
The ANN model used contained 3 Dense layers and 1 Output layer. The first 3 layers used sigmoid as the activation function
and the last layer used Mean Squared Unit as the activation function. 
The agent used here is a Supervised model which required labelled data for training.
We used the logic that the bird only needs to flap when its going to hit the bottom pipe in next frame, 
The model was trained after every 10000 steps for a total of ~200,000 steps. This model is able to beat the human top score of 1940.

---

<!--<b>Demo</b>-->

<!-- Add youtube video showing demo -->

<!--
---
-->

<b>Code Repository</b>

Click <a href="https://github.com/kasim95/Deep_Learning-Flappy_Bird.git" target="_blank">here</a> to view the Github repo.

---
