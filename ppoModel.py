import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 

from collections import deque
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, input_state = 10, action = 7):
        super(ActorNetwork, self).__init__()
        self.input_state = nn.Linear(input_state, 128)
        self.layer1 = nn.Linear(128, 128)
        self.action = nn.Linear(128, action)

    #neural network must have function for pytorch
    def forward(self, state):
        x = F.relu(self.input_state(state))
        x = F.relu(self.layer1(x))
        x = F.tanh(self.action(x))
        x = Categorical(x)
        return x
    
    #state-action value function, return the loss value 
    def loss_value(self, state_value, reward, next_state, gamma, done, loss_function):

        # if it's game over, win, or stall, which means there would be no next state
        # hence the difference
        if done:
           
            loss = loss_function(reward, state_value)
            #print("only reward")
            #print(reward)
            #print(state_value)
            #print(loss)
            loss.requires_grad_(True)
        else:
            next_state_value = self.forward(next_state)
            loss = loss_function(reward + gamma * torch.max(next_state_value), state_value)
           # print("not only reward")
           # print(reward)
            #print(state_value)
            #print(next_state_value)
           # print(loss)

        return loss
    
   
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, optimizer ):
        super(CriticNetwork,self).__init__()

        self.critic = nn.Sequential
        (
            nn.Linear(*input_dims, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optimizer
        
    def forward(self, state):
        value = self.critic(state)
        return value
    
    
class Trainer:
    def __init__(self, gamma, actor, critic, optimizer, loss_function, BATCH_SIZE = 100):
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer   
        self.loss_function = loss_function
        self.memory = []
        self.batch_size = BATCH_SIZE
    
    #don't work for now
    def save_model(self, file_name):
        model_folder_path= './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.model.dict(), file_name) 

    #don't work for now
    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name, weights))
        self.model.eval()

    #record for batch training
    def record_situation(self, state, action, reward, done, prob, val):
        self.memory.append((state, action, reward, done, prob, val))

    
    def select_batch(self):

       
        #select a random amount of sample if the sampl nubmer exceed the batch number
        if len(self.memory) > self.batch_size:
            random_samples = random.sample(self.memory, self.batch_size)
        
        else:
            random_samples = self.memory
        return random_samples
    
    def clear_memory(self):
        self.memory = []


        
    def train(self, state_value, reward, next_state, done):
        reward_tensor = torch.tensor(reward)
        next_state_tensor = torch.from_numpy(next_state)
        state_value_tensor = torch.from_numpy(state_value)

        #unsqueeze if the input is an array for better training
        #if len(state.shape) == 1:
        #    reward_tensor = torch.unsqueeze(reward_tensor, 0)
        #    next_state_tensor = torch.unsqueeze(next_state)
        #    state_value_tensor = torch.unsqueeze(state_value)

        loss = self.model.loss_value(state_value_tensor, reward_tensor, next_state_tensor, self.gamma, done, self.loss_function)
        #print(loss)
        loss.backward()
        self.optimizer.step()

    def batch_train(self):

        #get a random selected past results from the model
        samples = self.select_batch()
        samples_number = len(samples)

        #zip for better iteration
        states, actions, rewards, next_states, dones = zip(*samples)
        
        #do a training on every one of those samples for long memory
        
        for this_sample in range(samples_number):
            print("training for " + str(this_sample) + "right now.\n")
            self.train(actions[this_sample], 
                       rewards[this_sample], 
                       next_states[this_sample], 
                       dones[this_sample])
          



