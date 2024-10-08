import numpy as np
import random as rd
import matplotlib.pyplot as plt
import gym

alpha = 0.7 #learning rate                 
discount_factor = 0.618               
epsilon = 1                  
max_epsilon = 1
min_epsilon = 0.01         
decay = 0.01

train_episodes = 2000000
test_episodes = 100          
max_steps = 100

env = gym.make("Taxi-v3").env #render_mode=human or rgb_array

# print("Action Space {}".format(env.action_space))
# print("State Space {}".format(env.observation_space))

Q = np.zeros((env.observation_space.n, env.action_space.n))

training_rewards = []
epsilons = []
for episode in range(train_episodes):
    if episode % 10000==0:
        print(episode)
    state = env.reset()[0]
    total_training_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = rd.uniform(0,1) # Choosing an action
        if exp_exp_tradeoff > epsilon:
            # employing exploitation and selecting best action 
            action = np.argmax(Q[state, :])
        else:
            # Otherwise, employing exploration: choosing a random action
            action = env.action_space.sample()

        new_state, reward, done, _, info = env.step(action)

        Q[state, action] = Q[state, action] + alpha*(reward + discount_factor*np.max(Q[new_state, :]) - Q[state, action])

        total_training_rewards += reward
        state = new_state

        if done==True:
            break
        
        epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay*episode)

        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

print ("Training score over time: " + str(sum(training_rewards)/train_episodes))
      
