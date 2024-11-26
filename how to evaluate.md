# What are we evaluating : 
- algorithms 
    - Q-learning
    - Deep-Q learning
    - Gen algo: classical + NEAT

# What are the key components :
- time complexity
    - statistique
    - time to reach goal (finding cars that do at least one lap)
    - play with hyperparameters
- power usage 
- parallelisation 
- memory usage
- For GA: Markov gen modelisation -> no possible mixing time


autosuffisant 


# Structure of the report 

## Introduction 
- what is the project ?
présenter le pb , résumer ce que l'on vas faire, ce que l'on veut évaluer , méthodes

## Deep Q Learning 
- What is Markovian decision process ?
- What is Q Value ? 
- What is Q learning ? (Bellman equation) 
- What is Q learning ?

Optional :
- few words on alternatives to Q-learning 

## Car Racing environment 
- What is the physics of the car ?
- What are the main ideas/technical points behind the envorionment ?
- How do we compute reward ? Why this reward ?
- Why this features ?

## Complexity
- Estimate the complexity of the training and evaluation process
- model the training time 

## Experiments

- métriques d'un processus :
    - temps d'entrainement 
    - mesure de l'énergie 
    - es-ce que ça scale 
    - mesure de la mémoire 
    - mesure du reward moyen
    - combien de temps pour terminer un tour 
    - over fitting ?



- comparer avec vraie trajectoire 

Conclusion :
- limites , comment faire mieux 
