we want our function of training to return a dictionnary containing :

- res["training_time"] = training time allocated
- res["track_number"] = number of track used for training
- res["global_volatility"] = standard variation of the total reward of each episode
- res["model_size"] = number of parameters of the model
- res["policy_time"] = time for the model to compute the action
- res["policy_score"] = the average reward of the model after the training on the tracks    that are NOT used for the training
- res["reward_history"] = The reward observed at the end of each episode
- res["reward_time"] = The time at which we added each element of "reward_history"
- res["volatility_history"] = The standard variation of the last 30 reward observed , each time we add an element to reward_history
- res["volatility_time"] = same as "reward_time"
- res["DQN_model_param"] = filename of the file storing 
- res["DQN_model_param_is_saved"] = true or false depending on if we stored the parameters (only if the training last for at least 3 min)