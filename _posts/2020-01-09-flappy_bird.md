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


<b>Code:</b> <a href="https://github.com/kasim95/Deep_Learning-Flappy_Bird.git" target="_blank" style="color: black;">Github repo <i class="fab fa-github fa-lg"></i></a>

---

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
the obstacles in its path. The player controlled bird flies forward
and is pulled down by gravity. The player must tap the 
screen to flap wings and fly higher in order to avoid the pipes.

This project involved creating an agent that will be able to play
Flappy bird game in a team environment (Details of team members
can be found on the Github repository page).
The responsibility of the agent is to perform
the best action in order to move the bird forward avoiding any
obstacles in its path. 
The game environment used for this project 
is influenced by an 
existing implementation of Flappy Bird made using <i>pygame</i> in 
<i>Python</i> and available in this 
<a href="https://github.com/sourabhv/FlapPyBird.git">Github repository</a>.

<b>Deep Q Network</b>

On our <i>first</i> attempt, we used a Deep Q Network, a Reinforcement Learning 
algorithm as the agent to play the game. 
In Reinforcement Learning, an agent interacts with the environment 
arriving at different 
scenarios known as states by performing actions influenced by 
rewards while assuming that each state follows the 
<a href="https://en.wikipedia.org/wiki/Markov_model"><i>Markov model</i></a>.
The basic principle used in Q Learning is that at any given state, 
an action is performed that will eventually yield the 
highest cumulative reward.
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
which takes input as a stack of four consecutive frames of dimensions 80 x 80 px. 
The four frames were used to simulate the downward movement of bird.
The CNN model used contained 3 convolutional layers, 1 hidden layer and 1 output layer. 
The rectified linear unit (ReLU) activation function is used for the first 4 
layers and weighted output for the last layer.
To train the model, a memory store capable of holding 
game activity data for upto 50000 frames at a time is used. 
To collect enough data to train the model, the agent was run for 
50000 steps intially with a random action used 1/100 times.
<!-- Check this with epsilon value-->
After the intial 50000 runes, the model was trained every 
10000 steps with a batch size of 32 
for a total more than 1 million steps. 
For each frame, the memory contains the current frame as the image, 
action taken, reward obtained, and the next frame resulting from the 
action. 


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
