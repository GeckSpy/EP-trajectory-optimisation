Hi,
Here is our project proposal:


Our group for the project is: Mathieu Longatte, Louison Mocq and Macéo Ottavy
The main goal of our project is to optimize a trajectory of a simple 2D car model in a turn using AI.

Obejctives:
I) Modeling :
Modifying existing environment (and possibly creating our own) for our 2D car model using Gymnasium python package. If the other objectives are cleared, we could try to develop the model to optimize the trajectory of the car for an entire track. This would make us adapt our environment. We could also try to make our model more complex or we could add new aspect like obstacles.


II) Simulation :
 A) Implementation
  The purpose of using AI is to train the AI models to find a trajectory as good as possible then we will:
    - Implement Q-learning algorithm which is a algorithm to train an AI.
    - Implement Deep Q-learning with Pytorch python package.
    - May be implement other algorithms that we found or more basic one.
  In both case, we will have to play with the hyperparameters and the algorithm to make it as efficient as possible.

  B) Training
  - Train the AI in all the models with different training times and the algorithms (Q-learning VS Deep Q-learning VS other algorithm and/or more basic one).


III) Experimentation :
 We will analyze real trajectory of car racer to be able to compare the trajectories that our car make : Do IA models take turn like humans ? We could try to compare car racer’s trajectory and our trained model’s trajectory on a same track. We would surely compare our different models and also the trade off reward VS training time, between different algorithms, hyperparameters and hardware (CPU VS GPU).


 IV) Bonus
If we succeed to have good result on those previous objectives and if we still have enough time, we could try to optimize a far much harder things: the decision of our AI to be able to find good trajectories for random turns or random tracks. Then, we would have to rebuild entirely our environment and try to make heuristics or decision to solve our problem.



Respectfully,
Mathieu Longatte, Louison Mocq and Macéo Ottavy
