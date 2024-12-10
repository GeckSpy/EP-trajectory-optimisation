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
- Statistique
    - best reward for each iteration
    - average reward
- power usage 
- parallelisation 
- memory usage
- For GA: Markov gen modelisation -> no possible mixing time

- For best car found:
    - Average time to compute a track
    - average speed
    - Compare with real trajectory on some real circuit?


autosuffisant 


# Structure of the report 

## Introduction 
- what is the project ?
présenter le pb , résumer ce que l'on vas faire, ce que l'on veut évaluer , méthodes (summary of proposal)

## Deep Q Learning 
- What is Markovian decision process ?
- What is Q Value ? 
- What is Q learning ? (Bellman equation) 
- What is Q learning ?

Optional :
- few words on alternatives to Q-learning

## Genetic algorithm
- What is it?
- Markov chain modelisation

## Car Racing environment 
- What is the physics of the car ?
- What are the main ideas/technical points behind the envirionment ?
- How do we compute reward ? Why this reward ?
- Why this features ?

## Complexity
- Estimate the complexity of the training and evaluation process
- model the training time 

## Experiments

- métriques d'un processus :
    - temps d'entrainement 
    - mesure de l'énergie 
    - es-ce que ça scale : pas de discretisation
    - mesure de la mémoire 
    - mesure du reward moyen
    - combien de temps pour terminer un tour 
    - over fitting ?

- comparer avec vraie trajectoire 

Méthodologie :
- fixer un budget de temps :
    - temps court : 10min
    - temps long : 1h
- On mesure : 
    - cout de l'entrainement : 
        - temps : fixé a l'avance, pas besoin de le mesurer
        - énergie : utiliser PyJoule 
        - mémoire : ( nombre de paramètres du modèle)
    - performance du model : 
        - plot du reward par itération : X = temps , Y = reward à itération finissant au temps X
        - temps pour calculer la politique 
        - variance du reward par itération

How to evaluate the best car :
- 

Conclusion :
- limites, comment faire mieux 
