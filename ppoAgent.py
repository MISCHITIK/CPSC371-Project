import gymnasium as gym
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from minesweeper import Game
from ppoModel import ActorNetwork 
from ppoModel import CriticNetwork
from ppoModel import PPOTrainer
import matplotlib.pyplot as plt


gamma = 0.99
gae_lambda = 0.95
lr = 0.0003
policy_clip = 0.1 #epislon for the clipping in critic network or 0.2 for the clipping
n_epochs = 4

step = []
save_round = 10

board_width = 6
board_height = 6
number_of_mines = 2

total_step = 0
current_obs = torch.tensor
max_step_per_round = 2000
total_rounds = 3000 #epochs

score_history = np.array([])
game_state_history = np.array([])
best_score = 0

game = Game(board_width, board_height, number_of_mines)
actor = ActorNetwork(game.board_size, game.board_size, lr)
critic = CriticNetwork(game.board_size, lr)
Trainer = PPOTrainer(gamma, gae_lambda, actor, critic, n_epochs, lr, policy_clip)




#play for 50 rounds of games
for round in range(total_rounds):


    #reset the game, as well as get the initial observation space of the game
    current_obs = game.randomize_board()

    score = 0
    
    #max step per round is 999
    for step in range(max_step_per_round):   
            game.printGame()   
            #print(step)
            total_step += 1
           #print("current obs")
            #print(current_obs)
            #get necessary info to store state with certain action
            action, prob, val = Trainer.get_action(current_obs)
            print('current vals')
            print(val)

            #print('action')
            x, y = game.x_y_transform(action)
            
            next_obs, reward, done  = game.action(x, y)
            score += reward
            
            #record the situation into the memory
            Trainer.record_situation(current_obs, action, reward, done, prob, val)
            
            
            #current_obs = torch.from_numpy(next_obs)
            current_obs = next_obs
            #if it's done, it means can't play the game anymore, need to reset
            if done:
                score_history = np.append( score_history, [score])
                Trainer.batch_train()
                #record how many step in this round of play, for plotting
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
                break
            
    #save model on this certain round        
    #if round == save_round: Trainer.save_model() 
   
  

#print evaluation graph, number of steps taken
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
   