# 2024-10-28_18:04:26 

architecture : 2 24 relu - 24 relu - 4

adamw amsgrad true
lr = 1e-3
discount = 0.9

epochs 1000
batch size = 64 
replay memory = oui

gradient clipping


phase 1 :
min eps = 1
max eps 0.05
eps decay 1000

phase 2 : 
epsilon_max = 0.2
epsilon_min = 0.05
epsilon_decay = 500.0

resultat : parfait 

question : peut-on le faire en 1 run ?
abaissaer le temps de calcul ?


# update

mettre none pour un etat finis 

typé les fonctions et faire un descriptif

différencier appentissage et prediction : faire des petites transition 
alpla nouveau + (1-alpha) actuel

# pb de modélisation :

avant d'avoir bien estimé la qvalue , peut etre il vas préférer se suicider plutot que de se rapprocher 
car il trouve pas la sortie 

reinforcement learning , Sutton

# 12 novembre :

epochs = 2000
    batch_size = 1

    epsilon_max = 1
    epsilon_min = 0.02
    epsilon_decay = 300.
    lr = 1e-4
    discount_factor = 0.9

    optimizer = optim.AdamW(env.model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

explose le jeux : prends 20sec à s'executer 

