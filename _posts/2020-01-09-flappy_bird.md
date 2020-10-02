---
title: "Deep Learning - Flappy Bird"
date: 2020-01-09
tags: [deep learning, games, computer vision, artificial intelligence]
header: 
    overlay_color: "#ffffff"
    overlay_image: "/assets/images/bgs/good_old_times.png"
    overlay_filter: .4
    caption: "Photo Credit: [**Gijs de Wit**](http://simpledesktops.com/browse/desktops/2014/jun/08/good-old-times/)"
excerpt: "A Deep Learning agent trained to play Flappy Bird game"
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

<!-- todo: add demo videos and images and snippets of DQN and Brain classes. -->

<b>Tools Used</b>
<ul>
    <li>Python 3</li>
    <li>Tensorflow</li>
    <li>Keras</li>
    <li>Pygame</li>
</ul>


Flappy Bird game released in 2013 for iOS and Android mobile 
platforms instantly gaining popularity worldwide. 
The goal of the game is to fly the bird without hitting 
the obstacles in its path. The player controlled bird flies
forward is pulled down by gravity. The player must tap the 
screen to flap wings in order to fly higher and avoid the pipes.

The game environment used for this project is influenced on an 
existing implementation of Flappy Bird made using <i>pygame</i> in 
<i>Python</i> and available in this 
<a href="https://github.com/sourabhv/FlapPyBird.git">Github repository</a>.

<b>Deep Q Network</b>

On our <i>first</i> attempt, we used a Deep Q Network as the agent to play the game. 
The Deep Q Network is a Reinforcement Learning algorithm. 
In Reinforcement Learning, an agent interacts with an environment arriving at different 
scenarios known as states by performing actions influenced by rewards 
assuming each state follows the 
<a href="https://en.wikipedia.org/wiki/Markov_model"><i>Markov model</i></a>.
The basic principle used in Q Learning is that at any given state, an action is 
performed that will eventually yield the highest cumulative reward.
This Q-value is calculated using Bellman equation given as 

<i>Q(s, a) = r(s, a) + γ * max Q(s',a)</i> 

Here,

--- | ---
--- | ---
s | current state
i | action performed
r(s,a) |reward
s'| next state
max Q(s', a) | maximum Q-value for next state
γ | discount factor to control influence of future reward

For the Deep Q Network, we used a Convolutional Neural Network (CNN) model 
which takes input as a stack of four 80 X 80 px consecutive frames. 
The four frames were used to simulate the downward movement of bird.
The CNN model used contained 3 convolutional layers, 1 hidden layer and 1 output layer. 
The rectified linear unit (ReLU) activation function is used for the first 4 
layers and weighted output for the last layer.
To train the model, a memory able to hold game activity data 
for upto 50000 frames is used. 
To collect enough data to train the model, the agent was run for 50000 steps 
without training with a random action every 1 out of 100 steps and  
the action predicted by untrained model at other times.
<!-- Check this with epsilon value-->
The model was trained every 10000 steps with a batch size of 32 
for a total more than 1 million steps. 
Each cell in memory contained the current frame as the image, 
action taken, reward obtained, next frame resulting from the action 


<b>Artifical Neural network</b>

On our <i>second</i> attempt, a simple Multilayer Perceptron model as the 
agent to play the game.
This model used the bird's position in y-direction, the gap between bird &
top obstacle in y-direction, 
the distance between bird & upcoming pipe in x-direction 
and the downward velocity of bird in y-direction.
The ANN model used contained 3 Dense layers and 1 Output layer. 
The first 3 layers used sigmoid activation function
and the last layer used weighted output as the activation function. 
The agent used here is a Supervised model which required labelled data for training.
To generate labelled data, we used the logic that the bird only needs to flap 
when its going to hit the bottom pipe in next frame. 
The model was trained after every 10000 steps for a total of ~200,000 steps. 
This model is able to beat the human top score of 1940.

---

<b>Demo</b>

<img src="/assets/images/posts/flappy_bird-1.gif"/>

---

<b>Code Repository</b>

<a href="https://github.com/kasim95/Deep_Learning-Flappy_Bird.git" target="_blank">Click here</a> to view the Github repository

---
