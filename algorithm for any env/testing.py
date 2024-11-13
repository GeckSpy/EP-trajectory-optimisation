    
# def test_replay():
#     memory = ReplayMemory(3) # type: ignore
#     memory.push(Transition(1,2,3,4))
#     memory.push(Transition(11,12,13,14))
#     memory.push(Transition(111,112,113,114))
#     memory.push(Transition(1111,1112,1113,1114))
#     print(memory.memory)
#     print(len(memory))
#     print(memory.sample(1))
#     memory.clear()
#     print(len(memory))

# test_replay()


# def test_state() :
#     game_name = "CliffWalking-v0"
#     env= Env(game_name)
#     print(env.state())
#     print(env.step( torch.tensor(0)  ))
#     print(env.step( torch.tensor(1)  ))
#     print(env.step(  torch.tensor(0)))
#     print(env.step(  torch.tensor(2)))
#     print()
#     for i in range(11):
#         env.step( torch.tensor(1))
#     print(env.step(torch.tensor(2)))

# Testing 
# state, step, computation of reward , type of the state, size of the state, type of the action
# test_state()





# def test_sample_rand():
#     game_name = "CliffWalking-v0"
#     env= Env(game_name)
#     action = env.random_action()
#     print(action) 
#     print(env.step(action))

#     for i in range(1000):
#         env.reset()
#         while(not(env.done)):
#             env.step(env.random_action())

# test random policy 
# test_sample_rand()






# def test_policy():
#     game_name = "CliffWalking-v0"
#     env= Env(game_name)
#     print(env.model(env.state()))
#     print(env.policy())
#     for i in range(1000):
#         env.reset()
#         while(not(env.done)):
#             env.step(env.random_action())

#test_policy()



# # testing optimize
# # fill a batch with two transitions 
# env = Env("CliffWalking-v0")
# # env.replay.push( env.step( torch.tensor(0) ) )
# # env.replay.push( env.step( torch.tensor(1) ) )
# env.replay.push( env.step( torch.tensor(0) ) )
# env.replay.push( env.step( torch.tensor(1) ) )
# env.reset()
# env.replay.push( env.step( torch.tensor(1) ) )
# env.reset()
# env.replay.push( env.step( torch.tensor(0) ) )


# batch_size = 4
# transition = env.replay.sample(batch_size)
# batch = Transition(*zip(*transition))

# print(transition)

# state_batch = torch.cat(batch.state)
# action_batch = torch.cat(batch.action)
# reward_batch = torch.cat(batch.reward)


# predicted = env.model(state_batch).gather(1,action_batch)
# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool )
# print(non_final_mask)

# next_state_value = torch.zeros((batch_size,1))

# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool )
# non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
# next_state_value[non_final_mask] = env.model(non_final_next_state).max(1).values.unsqueeze(1)
# print(next_state_value)
