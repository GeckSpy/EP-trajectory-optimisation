# EP-trajectory-optimisation

Perfomance Evaluation project:

exemple of Q-learning with gym : https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d

why Q-learning converges ?  http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf

How do we do mathematical modeling ?
We don't have like precise idea on how the algorithms will behave, But we can give informal predictions on the convergences depending on the parameters based on the litterature on 
Q-learning algorithms , hyperparameters study (nb of epochs, batch_size, learning rate , depth of the network). 

Deep Q learning with Pytorch : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

gymnasium environment tutorial: https://medium.com/@ym1942/create-a-gymnasium-custom-environment-part-1-04ccc280eea9

DQN on car racing env : https://simmimourya.github.io/data/680_Report_RL.pdf

# Proposal:
2 parts:
What we are going to do
How we are going to do it

# The perfect env Class:
4 attribus :
- State : un tenseur représentant l'état
- Done : savoir si l'épisode est terminé
- observation_size : la dimension du tenseur qui représente les états
- action_size :  la dimension du tenseur qui représente les actions

Méthodes :
- Random_action State -> Action : échantillonne uniformément une action parmis les actions possibles étant donné un état
- Step State, Action -> Action, Reward : le reward doit être un flotant Torch (quelle taille?) , la fonction doit mettre à jour Done
