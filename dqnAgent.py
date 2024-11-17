import gymnasium as gym
import random

import numpy as np
import dqnModel
import torch.nn as nn
import torch.optim as optim
import torch
from minesweeper import Game
from dqnModel import DQNTrainer
from dqnModel import DQN
import matplotlib.pyplot as plt

memory_size = 100
batch_size = 8
gamma = 0.94
epsilon_decay_limit = 3000
lr = 0.00014
step = []
save_round = 10
board_width = 6
board_height = 6
mines = 2
#initialize game
game = Game(board_width,board_height,mines)
DQN = DQN(game.board_size,game.board_size)
Trainer = DQNTrainer(gamma, DQN, optim.Adam(DQN.parameters(), lr), nn.MSELoss(), batch_size, memory_size)


def x_y_transform( output_neuron_pos):
     x = output_neuron_pos % board_height
     y = output_neuron_pos % board_width

     return x, y


total_step = 0
current_obs = torch.tensor
total_rounds = 5000
reward = 0
done = True
state = 0
action = 0
best_score = 0
#play for 50 rounds of games
score_history = np.array([])
game_state_history = np.array([])
for round in range(total_rounds): 

    #reset the game, as well as get the initial observation space of the game
    current_obs = game.randomize_board()

    #set score
    score = 0
    
    for x in range(3):
        current_obs = game.reset_board()
        #max step is the board size + 10 for redundancy
        for step in range(999):      
            #print(step)
            total_step += 1

            #get an action, either random or generated through the network
            #epsilon, determines randomness, 
            # if above 0, make random moves, 
            # else feed it to neural network
            epsilon = random.random() - total_step/epsilon_decay_limit
            
            if(epsilon >= 0): 

                #print('random action')
                next_obs, action, reward, done = game.random_action()   #random action
                action_matrix = torch.zeros(game.board_size).to(torch.float32)
                action_matrix[action] = 1.0

            else: 
                #print('pass through network')
                state_value = DQN(current_obs)
                action = torch.argmax(state_value)
                x, y = x_y_transform(action)
                action_matrix = torch.zeros(game.board_size).to(torch.float32)
                action_matrix[action] = 1.0
                next_obs, reward, done = game.action(x, y)
                
             
            #step to get the necessary things for the training
            Trainer.record_situation(current_obs, action_matrix, reward, next_obs, done)
            
            #update current observation for the next round of training
            current_obs = next_obs

            #update the score for the current step
            score += reward


            #game.printGame()

            # if this round is finished, then start the batch training from the memory
            if done:
                score_history = np.append( score_history, [score])
                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                     best_score = avg_score
                     #Trainer.save_model()
                
                board_state = game.board_state()
                if(board_state == "Won"):
                     game_state_history = np.append(game_state_history,[1])
                elif(board_state == "Lost"):
                     game_state_history = np.append(game_state_history,[2])
                else: 
                     game_state_history = np.append(game_state_history,[3])

                print('played for ' + str(round) + ' round of games,'
                      +' this round played for ' + str(game.total_step) + ' steps. '
                      + 'The game is ' + str(board_state))     
                total_step = 0
                Trainer.batch_train()
                #record how many step in this round of play, for plotting
                
                break
     
            
#print graph


print(score_history)
print(range(1, len(score_history)))   
print("accuracy for the last 100 samples: " + str( np.count_nonzero(score_history[-100:] == 1) / len(score_history[-100: ])))


plt.plot(range(1, len( score_history[-100:])+ 1), score_history[-100: ])

plt.title('Data')

# Label the x-axis
plt.xlabel('Rounds')

# Label the y-axis
plt.ylabel('Scores')

plt.show()
   


#print evaluation graph, number of steps taken

