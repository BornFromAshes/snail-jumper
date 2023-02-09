# Snail jumper
The aim of this project is to use the evolutionary algorithm for neural network learning in an environment where there is not enough data for training. One of these environments is the game, where something new is always happening, and it is impossible to produce educational data for education. <br>

![Snail Jumber](SnailJumper.png)
<br>
The game designed in this project can be implemented in two modes, manual and neural development, and its goal is to cross the obstacles on the way, which is done in manual mode with space. After running the game.py file, you will see the following image, by choosing the first option, the game will run in manual mode, and by selecting the second option, you can run it in the form of neural evolution.

## Descirption
To run the game in the form of neural evolution, we need to design a neural network that receives important decision-making parameters as input and then produces the corresponding output. At the end, the generated output acts like pressing the space button.
![image](https://user-images.githubusercontent.com/117355603/217020595-bc987498-8913-40ff-9b6f-118d9a84b6d5.png)
Normally, to train a neural network, after determining the important parameters in choosing and building the architecture of the neural network, feedforward is done. Then a cost function should be defined so that in backpropagation, the weights and biases are updated in such a way that it leans towards the minimum. But in this project, there is no data for training and backpropagation, and that is why we use evolutionary algorithms. <br>
In this way, a large number of players are produced in the game (300 in that project), each of which has a neural network initialized with normal random weights and zero biases. Now, each of the players will show a different performance according to the available initial values, by observing the obstacles. The more the player continues on his path, the more fitness he acquires. According to the principle of evolution, the players with better performance will be transferred to the next generations, and by considering crossover and mutation after several generations, it is expected that they will show better performance and travel further.

## Project structure

- game.py : Implementation of the game process.
- evolution.py : Contains a class called evolution for the evolution of creatures of each generation.
- nn.py : Neural network architecture and feedforward section.
- player.py : Contains the player class to create the player(s) in the scene.
- variables.py : Contains public variables that are shared between files.

## Known Issues
In this project we used trial and error to find the best hyperparameters, activation functions and more. So if you find something that works better, feel free to create an issue on this repository

