import gymnasium as gym
import random

import numpy
import model
import torch.nn as nn
import torch.optim as optim
import torch
gamma = 0.94
epsilon_decay_limit = 3000
lr = 0.14
step = []
save_round = 10

DQN = model.DQN(2,1)
r_m = "human"

Trainer = model.Trainer(gamma, DQN, optim.Adam(DQN.parameters(), lr), nn.MSELoss(), 200, 10000)
#render_mode = "human" display the gameplay
env = gym.make("MountainCarContinuous-v0", render_mode = None)#apply_api_compatibility=True, render_mode="human"
#env = JoypadSpace(env, SIMPLE_MOVEMENT)


#display the action space
#print("action space: " + str(env.action_space))

#display observation space
#print("observation space:" + str(env.observation_space))


total_step = 0
current_obs = torch.tensor
max_step_per_round = 2000
total_rounds = 300
#play for 50 rounds of games
for round in range(total_rounds):
    if(round % 50 == 0): r_m = "human"
    else: r_m = None
    env = gym.make("MountainCarContinuous-v0", render_mode = r_m)
    #reset the game, as well as get the initial observation space of the game
    current_obs = env.reset()
    #somehow the env.reset returns a tuple of 3 elements, only the first one is an array
    #print(current_obs) to see the tuple
    current_obs = current_obs[0]

   

    #max step per round is 999
    for step in range(max_step_per_round):      
            #print(step)
            total_step += 1

            #get an action, either random or generated through the network
            #epsilon, determines randomness, 
            # if above 0, make random moves, 
            # else feed it to neural network
            epsilon = random.random() - total_step/epsilon_decay_limit
            
            if(epsilon >= 0): 
              
                 action = env.action_space.sample()   #random action
                 #print('random action\n' + 'action' + str(action))

            else: 
              
                obs_tensor = torch.from_numpy(current_obs)
                
                action_tensor = DQN(obs_tensor)              #feed it to the model
                action = action_tensor.detach().numpy()
                #print('use neural network'+'action' + str(action))
             
            #step to get the necessary things for the training
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            Trainer.record_situation(current_obs, action, reward, next_obs, done)
            

            #restart the game if the game is terminated(win or lose) or truncated(game in a stall)
            Trainer.train(current_obs, reward, next_obs , done)
            
            #depend on whether the action is different from the Q value of state, 
            #state might be required for the training input
            #as in train(state, action, reward, obs, done)
            
            #current_obs = torch.from_numpy(next_obs)
            current_obs = next_obs
            #if it's done, it means can't play the game anymore, need to reset
            if done:
                Trainer.batch_train()
                #record how many step in this round of play, for plotting
                
                break
            
    #save model on this certain round        
    #if round == save_round: Trainer.save_model() 
   
    #debug message           
    print('played for' + str(round) + 'of games, this round played for ' + str(total_step))         
env.close()

#print evaluation graph, number of steps taken

