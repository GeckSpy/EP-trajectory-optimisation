import os
import json
import neat
import time
import pickle
import numpy as np
import random as rd
from datetime import datetime
from env_06 import RacingCar, PATHS, TRACKS_FOLDER, TRACKS

env = RacingCar()
print(env.nb_state)
print("nb tracks:", len(TRACKS))


def save_genome(genome, path):
    f = open(path, 'wb')
    pickle.dump(genome, f)

def load_genome(path):
    f = open(path, "rb")
    genome = pickle.load(f)
    return genome

path_best_solution = "best_solution"

#save_genome(winner, path_best_solution)
#geno = load_genome(path_best_solution)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # To avoid overflow
    return e_x / e_x.sum()

def evaluate_genome(genome, config, nb_track_for_training, generation_act, show=False, id_track=None):
    # Create a neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Initialize the environment
    if id_track==None:
        id_track = rd.randint(0, nb_track_for_training-1)
    obs, _ = env.reset(TRACKS[id_track])
    total_reward = 0.0
    done = False

    max_step = 300
    step = 0
    while (not done) and (step < max_step):
        step += 1
        # Neural network decides the action
        if (generation_act < 3):
            action_probs = softmax(net.activate(obs))
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(net.activate(obs))  # Choose the action with the highest output

        obs, reward, done, _, _ = env.step(action)
        if show:
            env.render(show_trajectory=True)
        total_reward += reward
    
    return total_reward



def measure_policy_time(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    obs, _ = env.reset(TRACKS[rd.randint(0, len(TRACKS)-1)])
    time_deb = time.perf_counter()
    action = np.argmax(net.activate(obs))
    return time.perf_counter() - time_deb

def evaluate_model_reward(genome, config, track_budjet):
    rewards = []
    sum_rewards = 0
    for i in range(track_budjet, len(TRACKS)):
        rwrd = evaluate_genome(genome, config, 0, 100000, id_track=i)
        rewards.append(rwrd)
        sum_rewards += rwrd
    return sum_rewards/len(rewards)

def get_genomes_size(genome):
    num_nodes = len(genome.nodes)
    num_connections = len(genome.connections)
    total_size = num_nodes + num_connections
    #print(f"Genome Size: {total_size} (Nodes: {num_nodes}, Connections: {num_connections})")
    return total_size




def training(do_save=False,
             save_path="",
             time_bound=600,
             track_budjet=10,
             path_config="./RacingCar.config"):
    
    generation_act = 0
    liste_reward = []
    liste_reward_time = []

    def evaluate_population(genomes, config):
        for genome_id, genome in genomes:
            reward = evaluate_genome(genome, config, track_budjet, generation_act)
            genome.fitness = reward


    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        path_config)

    # Create the population
    population = neat.Population(config)
    # Add a reporter to display progress in the terminal
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    #winner = population.run(evaluate_population, n=NB_GEN)
    deb_time = time.perf_counter()
    while time.perf_counter() - deb_time < time_bound:
        population.run(evaluate_population, n=1)
        # Store the best genome of this generation
        best_genome = population.best_genome
        liste_reward.append(best_genome.fitness)
        liste_reward_time.append(time.perf_counter() - deb_time)
        #best_genome_per_gen.append((best_genome, best_genome.fitness))

        #print(f"Best genome of generation {generation}: {best_genome.fitness}")
        generation_act +=1


    dic_info = {}
    dic_info["training_time"] = time_bound
    dic_info["track_number"] = track_budjet
    dic_info["Config_path"] = path_config

    if do_save:
        save_genome(best_genome, save_path)
    dic_info["best_policy_path"] = save_path
    dic_info["is_saved"] = do_save

    dic_info["reward_history"] = liste_reward
    dic_info["reward_time"] = liste_reward_time
    dic_info["global_volatility"] = np.std(liste_reward)

    dic_info["model_size"] = get_genomes_size(best_genome)
    dic_info["policy_time"] = measure_policy_time(best_genome, config)
    dic_info["policy_score"] = evaluate_model_reward(best_genome, config, track_budjet)

    # Return the list of best genomes of each generation
    return dic_info

data_path = "../Return/data/data GA/"
path  = data_path + "run_part1_2024-12-05_16:02:41/" + "best_genome_3600sec_69tracks"
path2 = data_path + "run_part1_2024-12-05_22:38:16/" + "best_genome_21600sec_69tracks"
path3 = data_path + "run_part1_2024-12-06_08:49:23/" + "best_genome_21600sec_69tracks"
path4 = data_path + "run_part1_2024-12-06_14:46:15/" + "best_genome_3600sec_69tracks"
path5 = data_path + "run_part1_2024-12-06_14:48:17/" + "best_genome_3600sec_69tracks"
path6 = data_path + "run_part1_2024-12-06_14:48:58/" + "best_genome_25200sec_69tracks"
geno = load_genome(path5)

path_config = "./RacingCar.config"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        path_config)


evaluate_genome(geno, config, len(TRACKS), 10000, show=True)

def compute_data():
    liste_training_time_min = [10, 40, 60]# times in secondes
    liste_training_time_sec = [60*x for x in liste_training_time_min]

    nb_track_training = [40]#[8, 40, int(0.8*len(TRACKS))]

    folder_name =  "run_part1_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder_name)
    for nb_track in nb_track_training:
        for time_max in liste_training_time_sec:
            best_genome_path = folder_name + '/best_genome_' + str(time_max) + "sec_" + str(nb_track) + "tracks"
            json_object = json.dumps(training(time_bound=time_max, track_budjet=nb_track, do_save=True, save_path=best_genome_path))
            
            with open(folder_name + "/" + str(time_max) + "_" + str(nb_track) + ".json", "w") as f :
                f.write(json_object)


#compute_data()


