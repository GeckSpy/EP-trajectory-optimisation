# Essentials for training and env

# Car env
from env_06 import RacingCar, PATHS, TRACKS_FOLDER, TRACKS, MAX_SPEED

# For neural networks
import torch 
import torch.nn as nn 
import torch.functional as F
import torch.optim as optim

# For math computations
import numpy as np

# For random
import random as rd

# For envs
import gymnasium

# For time limit
import time

# structure to save transitions 
from collections import namedtuple , deque
Transition = namedtuple("Transition",["state","action","next_state","reward"])

# For plots
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import display

# For saving files
from datetime import datetime
import json
import os

# for model vizualisation 
from torchsummary import summary


class ReplayMemory():
    def __init__(self,maxlen : int):
        self.memory_ = deque(maxlen=maxlen)

    def push(self,x : Transition):
        self.memory_.append(x)

    def sample(self,batch_size : int) -> list[Transition]:
        return rd.sample(self.memory_,batch_size)
    
    def clear(self):
        return self.memory_.clear()
    
    def __len__(self):
        return len(self.memory_)


class DQN(nn.Module):
    def __init__(self,layer_size,state_size,action_n):
        super(DQN,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size,layer_size),
            nn.ReLU(),

            nn.Linear(layer_size,layer_size),
            nn.ReLU(),

            nn.Linear(layer_size,layer_size),
            nn.ReLU(),

            nn.Linear(layer_size,action_n),
        )

    def forward(self,x):
        return self.network(x)
    
    def save(self,filename : str = None):
        """save model's parameters in the given file"""
        if (filename == None):
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        torch.save(self.state_dict(),filename)

    def load(self,filename : str):
        """load parameters stored in the given file"""
        self.load_state_dict(torch.load(filename, weights_only=True))


class Env():
    def __init__(self):
        self.done = False

        # Underlying environment
        self.env = RacingCar()
        self.state_gym,_ = self.env.reset(TRACKS[0])
        self.n_action = self.env.nb_state

        # Current model estimating the Q-function
        self.model = DQN(400,8,self.n_action)

        # Transition history
        self.memory = ReplayMemory(10000)

        # To normalize the Q-function later, in the reward function
        self.discount_factor = 0.9

        # Number of tracks allowed
        self.track_for_training = int(len(TRACKS)*0.8)

    def state(self):
        """ On définit un état comme étant un batch de taille 1 ou None"""
        if (self.state_gym == None or self.done) :
            return None
        else :
            arr = np.array(self.env.get_state())
            arr = arr / max(MAX_SPEED, self.env.max_dist_wall) #To normalize the array
            return torch.tensor([arr],dtype=torch.float)

    def show_state(self):
        self.env.render(show_trajectory=True)

    def reset(self):
        """Reset the environment"""
        rd_track = rd.randint(0, self.track_for_training -1)
        self.state_gym , _ = self.env.reset(TRACKS[rd_track])
        self.done = False
    
    def dist(state):
        """Calcule la longueur d'un plus court chemin entre state et goal (sous forme d'un flotant)"""
        goal = torch.tensor([[11,3]],dtype=torch.float)
        start = torch.tensor([[0,3]],dtype=torch.float)
        if (torch.equal(state,start)):
           return torch.tensor(13,dtype=torch.float)
        else :
           return torch.sum(torch.abs(state-goal))
       
    def step(self,action : torch.tensor) :
        """ Fais un pas depuis l'état actuel via l'action donnée et renvoit la transition observéex
            Une action est un tenseur contenant un seul scalaire """
        if (self.done):
            raise(ValueError("Trying to move from a final state"))

        prev_state = self.state()

        # do the step and update the new gym state
        acc, turn = self.env.int_to_action(action.item())
        if np.absolute(turn) > self.env.max_turn:
            print(action.item(), (acc, turn))
            
        self.state_gym,reward,terminated,truncated,_ = self.env.step(action.item())
        self.done = terminated or truncated

        next_state = self.state()

        reward_normalizer = self.env.reward_max
        qtable_normalizer = 1/(1-self.discount_factor)
        reward = torch.tensor(reward/(reward_normalizer*qtable_normalizer), dtype=torch.float).reshape((1,1))
        action = torch.tensor(action.item()).reshape((1,1))

        transition = Transition(prev_state, action, next_state , reward)
        return transition
    
    def policy(self):
        if (self.done):
            raise(ValueError("Trying to predict a move from a final state"))
        return self.model(self.state()).max(1).indices.reshape((1,1))
    
    def random_action(self) -> torch.tensor :
        if (self.done):
            raise(ValueError("Trying to sample a move from a final state"))
        action = rd.randint(0,self.n_action-1)
        return torch.tensor(action).reshape((1,1))

def optimize(env : Env,optimizer,criterion,batch_size,discount_factor):
    if (len(env.memory) < batch_size) :
        return 

    # A list of batch_size transtions
    transition = env.memory.sample(batch_size)

    # A tuple with four coordinates : 
    # state -> a batch of size batch_size of states 
    # action -> a batch of size batch_size of actions
    # ect
    batch = Transition(*zip(*transition))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Batch of size batch_size of the Qvalue predicted by our current model, for the state and action of a transtion
    predicted = env.model(state_batch).gather(1,action_batch)

    next_state_value = torch.zeros((batch_size,1))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool )
    if non_final_mask.any():
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
        with torch.no_grad():
            next_state_value[non_final_mask] = env.model(non_final_next_state).max(1).values.unsqueeze(1)

    expected = reward_batch + (discount_factor * next_state_value)

    optimizer.zero_grad()
    loss = criterion(predicted,expected)
    loss.backward()
    torch.nn.utils.clip_grad_value_(env.model.parameters(), 100)
    optimizer.step()


def measure_policy_time(env) :
    env.reset()
    time_deb = time.perf_counter()
    env.policy()
    return time.perf_counter() - time_deb

def measure_model_size(env):
    return sum(p.numel() for p in env.model.parameters())


def evaluate_model_reward(env):
    sum = 0
    max_step = 300
    for i in range(env.track_for_training,len(TRACKS)) :
        env.state_gym , _ = env.env.reset(TRACKS[i])
        env.done = False
        i = 0
        while(i < max_step and not(env.done)) :
            i+=1
            transition = env.step( env.policy() )
            sum += transition.reward.item()
    return sum/( len(TRACKS) - env.track_for_training)

def training(lr=1e-4,epsilon_decay=30.,batch_size = 40,time_bound = 60*(1),track_budget=int(0.8*len(TRACKS))):
    env = Env()
    filename =  "saved_model/"  + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # track budget for training
    env.track_for_training = track_budget

    # Hyperparameters
    #batch_size = 40
    epochs = 5000
    max_episode_duration = 1000 * 1/env.env.time
    epsilon_max = 1
    epsilon_min = 0.01
    #epsilon_decay = 30.
    #lr = 1e-4
    discount_factor = 0.9
    env.discount_factor = discount_factor
    optimizer = optim.AdamW(env.model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    env.memory.clear()

    reward_history = []
    reward_time = []
    volatility_history = []
    volatility_time = []

    #time_bound = 60*(1)
    time_start = time.perf_counter()
    i = 0
    while ( (time.perf_counter() - time_start <= time_bound)  ):
        i += 1
        env.reset()
        epsilon = epsilon_min + (epsilon_max-epsilon_min)*np.exp(-i/epsilon_decay)
        it_counter = 0
        reward = 0
        while(not(env.done) and it_counter < max_episode_duration):
            it_counter += 1
            # Chose an action
            if (rd.random() <= epsilon):
                action = env.random_action()
            else:
                with torch.no_grad() :
                    action = env.policy()

            # Apply the transition and save it in memory
            transition = env.step(action)
            reward += (transition.reward).item()
            env.memory.push(transition)
            # Optimize by observing batch_size random transitions
            optimize(env,optimizer,criterion,batch_size,discount_factor)
        
        # Stats about the training
        second = (int(time.perf_counter() - time_start)) % 60
        minute = (int(time.perf_counter() - time_start)) //60
        # We save the model every 5 minutes
        if (minute%5 == 0 and minute > 3) :
            env.model.save(filename)
        normalizer = 1
        window_len = 30
        iteration_time = time.perf_counter() - time_start
        reward_time.append( iteration_time  )
        volatility_time.append(iteration_time)

        reward_history.append(reward*normalizer)
        
        last_window = reward_history[-window_len:]
        volatility_history.append(np.std(last_window))
        
    
    res = {}
    res["training_time"] = time_bound
    res["track_number"] = len(TRACKS)
    res["global_volatility"] = np.std(reward_history)
    res["model_size"] = measure_model_size(env)
    res["policy_time"] = measure_policy_time(env)
    res["policy_score"] = evaluate_model_reward(env)
    res["reward_history"] = reward_history
    res["reward_time"] = reward_time 
    res["volatility_history"] = volatility_history
    res["volatility_time"] = volatility_time 
    res["DQN_model_param"] = filename
    res["DQN_model_param_is_saved"] = False
    res["learning rate"] = lr 
    res["batch size"] = batch_size
    res["epsilon decay"] = epsilon_decay

    # we save the model if we trained it for more that 3 minutes
    minute = (int(time.perf_counter() - time_start)) //60
    if (minute > 3) :
         env.model.save(filename)
         res["DQN_model_param_is_saved"] = True

    # plt.plot(reward_history)
    # plt.plot(volatility_history)
    # plt.show()
    
    return res


def playing(env):
    sum = 0
    max_step = 300
    mini = 0
    for i in range(mini,len(TRACKS)) :
        env.state_gym , _ = env.env.reset(TRACKS[i])
        env.done = False
        j = 0
        while(j < max_step and not(env.done)) :
            j+=1
            transition = env.step( env.policy() )
            sum += transition.reward.item()
            env.show_state()
    return sum/( len(TRACKS) - mini)



def part1_data():
    training_times_min = [10,40,60]
    training_times_sec = [60*i for i in training_times_min]
    track_limit = [8,40, int(0.8*len(TRACKS)) ]
    if (len(TRACKS) < max(track_limit)) :
        raise(ValueError("not enough tracks"))
    folder_name =  "run_part1_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder_name)
    for t in training_times_sec :
        for n_track in track_limit :
            json_object = json.dumps( training(time_bound=t,track_budget=n_track))
            with open(folder_name + "/" + str(t)+"_"+str(n_track)+".json","w") as f :
                f.write(json_object)

def part2_data():
    training_time = 30*(60)
    lr_l = [1e-2,1e-3,1e-4,1e-5,1e-6]
    batch_size_l = [10,30,50,80,120]
    epsilon_decay_l = [10.,30.,50.,120.,200.]
    track_limit = int(0.8*len(TRACKS))

    folder_name =  "run_part2_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder_name)

    for par in lr_l :
        json_object = json.dumps( training(lr=par, track_budget=track_limit , time_bound=training_time ))
        with open(folder_name + "/" + "lr"+"_"+str(par)+".json","w") as f :
            f.write(json_object)

    for par in batch_size_l :
        json_object = json.dumps( training(batch_size=par, track_budget=track_limit , time_bound=training_time ))
        with open(folder_name + "/" + "batch_size"+"_"+str(par)+".json","w") as f :
            f.write(json_object)

    for par in epsilon_decay_l :
        json_object = json.dumps( training(epsilon_decay=par, track_budget=track_limit , time_bound=training_time ))
        with open(folder_name + "/" + "epsilon_decay"+"_"+str(par)+".json","w") as f :
            f.write(json_object)


def part3_data():
    training_times = (60)*(60)*6
    track_limit = int(0.8*len(TRACKS))
    folder_name =  "run_part3_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder_name)
    json_object = json.dumps( training(time_bound=training_times,track_budget=track_limit, lr=1e-5 , epsilon_decay=500.))
    with open(folder_name + "/" + "long training"+".json","w") as f :
        f.write(json_object)
            

# part1/2/3_data are used to compute the data of the section Evaluation of the report

