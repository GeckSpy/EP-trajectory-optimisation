# Gymnasium environment informations

### Important function for environment
 - reset() : Resets the environment to an initial state, required before calling step.
             Returns the first agent observation for an episode and information, i.e. metrics, debug info.

 - step()  : Updates an environment with actions returning the next agent observation, the reward for taking that actions,
             if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.

 - render(): Renders the environments to help visualize what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text.

 - close() : Closes the environment, important when external software is used, i.e. pygame for rendering, databases.

 - get_space() : is necessary for Mathieu's deep Q-learning algo


### Inheritancy
We inherit the "Env" class of Gymnasium and redefine these 4 functions to have access to already existing function and good compatibility.


### During initialization, we define several critical aspects:
 - Action space: A set of all possible actions that an agent can take in the environment.
        It’s a way to outline what actions are available for the agent to choose from at any given step

 - Observation space: A size or shape of the observations that the agent receives from the environment.
        Essentially, it describes the form and structure of the data the agent uses to make decisions.

 - Initial state: A starting state of the agent when the environment is initialized.
 