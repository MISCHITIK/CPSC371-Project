import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math 
import numpy as np

from collections import deque
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, input_state = 10, action_space = 7, alpha = 0.97, chkpt_dir = 'tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_state, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128, action_space)
            
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torvch_ppo')


    #neural network must have function for pytorch
    def forward(self, state):
        x = self.actor(state)

        x = Categorical(F.sigmoid(x))
        #print('actor categorical distribution')
        #print(x)
        return x
    
 
        
    def save(self):
        T.save(self.state_dict(), self.checkpoint_file)
    

    def load(self):
        self.load_state_dic(T.load(self.checkpoint_file))
   
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, chkpt_dir = 'tmp/ppo'):
        super(CriticNetwork,self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.ReLU(True)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

    def forward(self, state):
        value = self.critic(state)
        return value
    


    def save(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load(self):
        self.load_state_dic(T.load(self.checkpoint_file))
   


class PPOTrainer:
    def __init__(self, gamma, gae_lambda, actor, critic,  n_epochs, lr, policy_clip = 0.1):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.policy_clip = 0.1
        self.actor = actor
        self.critic = critic  

        self.n_epochs = n_epochs 
        self.memory = []
    
        
    
    #don't work for now
    def save_model(self, file_name):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

   
    #don't work for now
    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name, weights))
        self.model.eval()

    #record for batch training
    def record_situation(self, state, action, reward, done, prob, val):
        self.memory.append((state, action, reward, done, prob, val))

    
    def clear_memory(self):
        self.memory = []


    def get_action(self, obs):
       
        #distribution
        dist = self.actor(obs)
        value = self.critic(obs)
        action = dist.sample()

  
        probs = T.squeeze(dist.log_prob(action)).item()
        probs = T.tensor([probs])

        action = T.squeeze(action).item()
        action = T.tensor([action])

        value = value.item()
        value = T.tensor([value])

        return action, probs, value
    

    def get_shuffled_memory(self):

        memory = random.shuffle(self.memory)
       
    
    #basically state-value action, except it is a aggregate 
    def get_advantage(self,rewards, vals, dones):


        advantage = []

     
        #get advantage
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
          
                 a_t += discount*(rewards[k] 
                                  + self.gamma*vals[k+1] * (1-int(dones[k])) 
                                  - vals[k])
                 discount *= self.gamma* self.gae_lambda
            advantage.append(a_t)
        
        #for the last element
        adv = rewards[-1] - vals[-1]
        advantage.append(adv)

        return advantage
    
    def get_critic_values(self, states):
        critic_values = []
        for state in states:
            critic = self.critic(state)
            critic_values.append(critic)
        
        return critic_values
        
    def train(self, state, action, reward, done, old_prob_actor, old_critic_value, advantage):

        
        #actor loss
        dist = self.actor(state)               
        new_prob_actor = dist.log_prob(action)
        prob_ratio = new_prob_actor.exp()/ old_prob_actor.exp()
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage
        actor_loss = T.min(weighted_probs, weighted_clipped_probs).mean()

        #critic loss

        #new critic value
        new_critic_value = self.critic(state)
        new_critic_value = T.squeeze(new_critic_value)

        returns = advantage + old_critic_value
        critic_loss = (returns- new_critic_value)**2
        critic_loss = critic_loss.mean()


        #backward propagation and training of both model
        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()


    
    def batch_train(self):
        batches = []
        number_of_items = len(self.memory)
        not_empty = True
        batch_size = math.floor(number_of_items / self.n_epochs) 
        if(batch_size == 0): batch_size = 1
        

        #for each epoches
        while(not_empty):
            if len(self.memory) >= batch_size:
                batches = self.memory[0: batch_size]
                self.memory = self.memory[batch_size:]
            else:
                batches = self.memory
                break


            #zip for better iteration
            states, actions, rewards, dones, probs, vals = zip(*batches)
            
            
            #get advantage
            advantages = self.get_advantage(rewards, vals, dones)

            print('advanges')
            print(advantages)

            #print('reward')
            #print(len(rewards))

            #print(len(advantages))
            #do a training on every one of those samples 
            for this_sample in range(len(batches)):
                #print("training for " + str(this_sample) + "right now.\n")
                self.train(states[this_sample],
                       actions[this_sample], 
                       rewards[this_sample], 
                       dones[this_sample],
                       probs[this_sample],
                       vals[this_sample],
                       advantages[this_sample])
           
        #clear memory on the end of epoch
        self.clear_memory()


