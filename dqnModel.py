import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 

from collections import deque

class DQN(nn.Module):
    def __init__(self, input_state = 10, action = 7):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_state, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128, action),
           
        )

    #neural network must have function for pytorch
    def forward(self, state):
        #print("state before passing to neural network")
        #print(state)
        x = F.sigmoid(self.layers(state))
        #print("after third layer, giving out output\n\n")

        
        return x
    

    
class DQNTrainer:
    def __init__(self, gamma, model, optimizer, loss_function, BATCH_SIZE = 100, MAX_SAMPLE = 10000):
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer   
        self.loss_function = loss_function
        self.memory = deque(maxlen = MAX_SAMPLE)
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
    def record_situation(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def select_batch(self):

       
        #select a random amount of sample if the sampl nubmer exceed the batch number
        if len(self.memory) > self.batch_size:
            random_samples = random.sample(self.memory, self.batch_size)
        
        else:
            random_samples = self.memory


        return random_samples
    
        #state-action value function, return the loss value 
    def loss_value(self,current_state, action_matrix, reward, next_state, gamma, done, loss_function):

        # if it's game over, win, or stall, which means there would be no next state
        # hence the difference
        state_value = self.model.forward(current_state)
        state_value = torch.max(state_value)
        if done:
            #print("no next state")
            #print(reward)
            #print("current_state")
            #print(current_state)

            loss = loss_function(reward, state_value).to(torch.float32) 
     
            #print("loss: ", loss ,"\n\n")

            loss.requires_grad_(True)
        else:
            #print("have next state")
            #print(reward)
            #print("next state ")
            #print(next_state)
            next_state_value = self.model.forward(next_state)
            next_state_value = torch.max(next_state_value)
            loss =  loss_function(reward + gamma * next_state_value, state_value) 
            loss.requires_grad_(True)
            #print(state_value)
            #print(next_state_value)
            #print("loss: ", loss, "\n\n")

    
        #loss = loss.to(torch.float32)
        return loss



    def train(self, state, action_matrix, reward, next_state_tensor, done):
        reward_tensor = torch.tensor(reward).to(torch.float32)
        #print("action_matrix")
        #print(action_matrix)


        #unsqueeze if the input is an array for better training
        #if len(state.shape) == 1:
        #    reward_tensor = torch.unsqueeze(reward_tensor, 0)
        #    next_state_tensor = torch.unsqueeze(next_state)
        #    state_value_tensor = torch.unsqueeze(state_value)

        loss = self.loss_value(state, action_matrix, reward_tensor, next_state_tensor, self.gamma, done, self.loss_function)
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
            #print("training for " + str(this_sample) + " situation in memory right now.\n")
            self.train(states[this_sample],
                       actions[this_sample], 
                       rewards[this_sample], 
                       next_states[this_sample], 
                       dones[this_sample])
          



