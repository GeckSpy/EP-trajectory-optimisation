import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random as rd
import time
import os
import json
import neat
import sys
import pickle

from gym import Env, spaces
from datetime import datetime
from os import walk


class Coor():
    def __init__(self, coor):
        self.x = coor[0]
        self.y = coor[1]

    def get(self):
        return self.x, self.y
    
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    
    def __add__(self, coor2):
        return Coor((self.x + coor2.x, self.y + coor2.y))
    
    def __eq__(self, coor2):
        if coor2 == None:
            return False
        return (self.x==coor2.x) and (self.y==coor2.y)
    
    def __neg__(self):
        x,y = self.get()
        return Coor((-x,-y))
    
    def __sub__(self, coor2):
        coor = - coor2
        return self + coor
    
    def norm(self):
        x,y = self.get()
        return np.sqrt(x*x + y*y)
    
    def dist(self, coor2):
        return (self -coor2).norm()


def intersect(coorA,coorB,coorC,coorD):
    # Return true if line segments AB and CD intersect
    def ccw(coorA, coorB, coorC):
        return (coorC.y-coorA.y) * (coorB.x-coorA.x) > (coorB.y-coorA.y) * (coorC.x-coorA.x)
    return ccw(coorA,coorC,coorD) != ccw(coorB,coorC,coorD) and ccw(coorA,coorB,coorC) != ccw(coorA,coorB,coorD)


RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
GREY = [70 for _ in range(3)]
WHITE = [240 for _ in range(3)]

START_CHAR = 2
CAR_CHAR = 4

def color_track(b):
    if b == START_CHAR:
        return GREEN
    elif b == 1:
        return GREY
    else:
        return WHITE
    

class Track():
    def __init__(self, tab, l_bt_lines=8, nb_lines=1, compute_lines=True):
        """ Track class
        
        l_bt_lines is the space between two lines

        nb_lines is tej inverse of the ratio you want to keep each lines
        i.e. nblines=n => keep on 1/n lines
        """

        #switching height and width for plan approach
        self.height, self.width = np.array(tab).shape
        self.basic_info_track:list = np.array(tab)
        
        self.info_track:list = [[0 for _ in range(self.height)] for _ in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                self.info_track[x][y] = self.basic_info_track[self.height-1-y][x]

        self.color_track = [[color_track(self.info_track[x][y]) for x in range(self.width)] for y in range(self.height)]

        self.start = None
        for y in range(self.height):
            for x in range(self.width):
                if self.info_track[x][y] == START_CHAR:
                    self.start = Coor((x,y))

        self.basic_alpha = 0
    
        self.nb_lines = nb_lines
        self.lenght_bt_lines = l_bt_lines

        self.midpoints = []
        self.lines = []
        if compute_lines:
            self.midpoints, self.lines = self.create_lines(self.start)
            self.lines = [x for i,x in enumerate(self.lines) if i%nb_lines==0]
        

    def create_lines(self, coor:Coor):
        liste_coor = [coor]
        lines = []
        alpha = 0

        running = True
        while True:
            line, next_alpha, new_coor = self.create_line(liste_coor[-1])
            #plt.plot(liste_coor[-1].x, liste_coor[-1].y, "o", color="limegreen", markersize=3)
            liste_coor[-1] = new_coor

            coor_alpha = Coor((np.cos(alpha*np.pi/180), np.sin(alpha*np.pi/180)))
            coor_next_alpha = Coor((np.cos(next_alpha*np.pi/180), np.sin(next_alpha*np.pi/180)))
            if coor_alpha.x*coor_next_alpha.x + coor_alpha.y*coor_next_alpha.y<0:
                next_alpha = (next_alpha + 180) %360

            dx = np.cos(next_alpha*np.pi/180) * self.lenght_bt_lines
            dy = np.sin(next_alpha*np.pi/180) * self.lenght_bt_lines
            next_coor = Coor((dx,dy)) + liste_coor[-1]
            

            liste_coor.append(next_coor)
            lines.append(line)
            alpha = next_alpha
            
            if (not running) and intersect(new_coor, next_coor, lines[0][0], lines[0][1]):
                break
            running = False
            
        return liste_coor, lines


    def create_line(self, base_coor):
        min_lenght = self.height + self.width
        for alpha in range(0, 180):
            coor1_act = self.next_wall(base_coor, alpha)
            coor2_act = self.next_wall(base_coor, alpha+180)
            lenght_act = coor1_act.dist(coor2_act)

            if lenght_act<min_lenght:
                coor1 = Coor((coor1_act.x, coor1_act.y))
                coor2 = Coor((coor2_act.x, coor2_act.y))
                min_lenght = lenght_act
                basic_alpha = alpha-90

        return (coor1, coor2), basic_alpha, Coor(((coor1.x + coor2.x)/2, (coor1.y + coor2.y)/2 ))

    def get_color(self, coor:Coor):
        """return the color of the case x,y"""
        x,y = coor.get()
        return color_track(self.info_track[x][y])
    
    def is_wall(self, coor:Coor):
        """Return True if case (x,y) is a wall"""
        x,y = coor.get()
        nx,ny = int(round(x)), int(round(y))
        return (self.info_track[nx][ny] == 1)

    def get_start(self):
        """Return coordinate of start"""
        if self.start == None:
            return None
        return self.start.get()
    
    def get_end(self):
        """Return coordinate of end"""
        return self.end.get()
    
    def is_case_ridable(self, coor: Coor):
        """Return if the car can go on the coordinate or not"""
        x,y = coor.get()
        x,y = int(round(x)), int(round(y))
        if not (x>=0 and x<self.width and y>=0 and y<self.height):
            return False
        return not self.is_wall(coor)
    
    def is_move_possible(self, a:Coor, b:Coor) -> bool:
        """Return if the car can go from point a to b in straight line"""
        diff_x = b.x-a.x
        diff_y = b.y-a.y

        d = a.dist(b)
        if d<1:
            d = 1
        
        space = np.arange(0, 1, 1/d)
        for t in space:
            case = Coor((a.x+t*diff_x, a.y+t*diff_y))
            if not self.is_case_ridable(case):
                return False
        return True
    
    def is_case_in(self, coor:Coor):
        """return True is coor is in the tab"""
        return coor.x>=0 and coor.x<self.width and coor.y>=0 and coor.y<self.height
    
    def next_road(self, coor:Coor, alpha:float, dist_max=None):
        """Return the next in the line from coor to the first wall"""
        alpha = alpha % 360
        dx = np.cos(alpha * np.pi/180)
        dy = np.sin(alpha * np.pi/180)

        i = 0
        next_coor = Coor( (int(round(coor.x + i*dx)), int(round(coor.y + i*dy))) )
        while not self.is_case_ridable(next_coor):
            if ((dist_max!=None) and (coor.dist(next_coor) > dist_max)) or (not self.is_case_in(next_coor)):
                return None
            i += 1
            next_coor  = Coor( (int(round(coor.x + i*dx)), int(round(coor.y + i*dy))) )
        return next_coor

    def next_wall(self, coor:Coor, alpha:float, dist_max=None):
        """Return the next in the line from coor to the first wall"""
        alpha = alpha % 360
        dx = np.cos(alpha * np.pi/180)
        dy = np.sin(alpha * np.pi/180)

        i = 0
        next_coor = Coor( (int(round(coor.x + i*dx)), int(round(coor.y + i*dy))) )
        while self.is_case_ridable(next_coor):
            if (dist_max!=None) and (coor.dist(next_coor) > dist_max):
                break
            i += 1
            next_coor  = Coor( (int(round(coor.x + i*dx)), int(round(coor.y + i*dy))) )
        return next_coor

    def plot(self, hide=False, show_lines=False, show_midpoints=False):
        """Plot the track using matplotlib"""
        plt.imshow(self.color_track, origin='lower')

        if show_lines:
            for i in self.lines:
                liste_x = [coor.x for coor in i]
                liste_y = [coor.y for coor in i]
                plt.plot(liste_x, liste_y, '-', color="lightblue")

        if show_midpoints:
            for i in self.midpoints:
                plt.plot(i.x, i.y, 'o', color="lightblue", markersize=3)

        plt.plot(self.start.x, self.start.y, "o", color="limegreen")
            
        plt.axis("off")
        if not hide:
            plt.show()


from matplotlib.image import imread
from PIL import Image

def info_from_real_color(tab):
    x,y,z = tab[0], tab[1], tab[2]
    if x==0 and y==0 and z==0:
        return 1
    elif np.sqrt((x-255)**2 + (y-255)**2 + (z-255)**2) <= 25:
    #elif x==255 and y==255 and z==255:
        return START_CHAR
    else:
        return 0
    
def crop(tab):
    start = None
    for i,x in enumerate(tab):
        for j,y in enumerate(x):
            if y==START_CHAR:
                start = Coor((i,j))
    mini = Coor((start.x, start.y))
    maxi = Coor((start.x, start.y))

    for i,x in enumerate(tab):
        for j,y in enumerate(x):
            if y==0:
                if i<mini.x:
                    mini.x = i
                if j<mini.y:
                    mini.y = j
                if i>maxi.x:
                    maxi.x = i
                if j>maxi.y:
                    maxi.y=j
    
    k = 2
    res = [[y for j,y in enumerate(x) if mini.y-k<=j<=maxi.y+k] for i,x in enumerate(tab) if mini.x-k<=i<=maxi.x+k]
    return res

def create_track_info(path):
    img = Image.open(path)
    arr = np.array(img)
    img.close()
    return crop([[info_from_real_color(y) for y in x] for x in arr])

TRACKS_FOLDER = "./tracks2/post_images/"
LINES_FOLDER = "./track_lines/"

PATHS = []
for (dirpath, dirnames, filenames) in walk(TRACKS_FOLDER):
    for file in filenames:
        file_path = TRACKS_FOLDER + file
        PATHS.append((file_path, file[:-4]))
    break

PATHS.sort()
# Tracks to del: 29, 40, 49, 65, 84, 92
to_del = [29, 40, 49, 65, 84, 92]
to_del.sort(key=lambda x:-x)
for x in to_del:
    del PATHS[x-1]


NB_PATH = len(PATHS)
rd.shuffle(PATHS)
PATHS = PATHS[:NB_PATH]


print("Number of tracks:", len(PATHS))
print(PATHS)


def save_lines(track:Track, path:str):
    file = open(path, "w")
    for c1,c2 in track.lines:
        txt = str(c1.x) + "," + str(c1.y) + "," + str(c2.x) + "," + str(c2.y) +"\n" 
        file.write(txt)
    file.close()


def save_tracks_lines():
    # put all the tracks path here:
    # The track called O3.png is not working because of unsmooth corner
    for name, number in PATHS:
        track_info = create_track_info(name)
        track = Track(track_info, nb_lines=2, l_bt_lines=8)
        save_lines(track, LINES_FOLDER + number + ".txt")
        print(name, number)


def create_tracks():
    tracks = []
    for name, number in PATHS:
        track_info = create_track_info(name)
        track = Track(track_info, nb_lines=2, l_bt_lines=8, compute_lines=False)
        file = open(LINES_FOLDER + number + ".txt", "r")

        for lines in file:
            a,b,c,d = lines.split(",")
            a,b,c,d = int(a), int(b), int(c), int(d)
            c1 = Coor((a,b))
            c2 = Coor((c,d))
            track.lines.append((c1,c2))
            track.midpoints.append(Coor(((c1.x+c2.x)/2, (c1.y+c2.y)/2)))
        file.close()
        tracks.append(track)

        #tracks[-1].plot(show_lines=True, show_midpoints=True)
        print(name, "as", len(tracks[-1].lines), "lines")

    return tracks


#save_tracks_lines() # uncomment to not recalculate lines

TRACKS:list[Track] = create_tracks()

"""Constant"""
MAX_SPEED = 50
MAX_TURN = 20

"""Class"""
class Car():
    def __init__(self, coor:Coor, time):
        self.coor: Coor = Coor((coor.x, coor.y))
        self.speed: float = 0
        self.alpha: float = 0 # The angle of the car according to unitary cicrle
        self.trajectory = [[Coor((coor.x, coor.y)), 0]]
        self.previous_speed: float = 0
        self.time = time

        self.max_turn = 20 * self.time
        self.max_speed = 50 * self.time
        self.acceleration_constant = 3 * self.time
        self.brake_constant = 6 * self.time

    def __str__(self):
        return "C[" + str(self.coor) + " " + str(self.speed) + " " + str(self.alpha) + "]"
    

    def accelerate(self, amont=1):
        """Increase speed of the car"""
        self.speed += amont * self.acceleration_constant
        self.speed = min(self.speed, self.max_speed)
        

    def brake(self, amont=1):
        """Decrease speed of the car (can't drive backward)"""
        self.speed -= amont * self.brake_constant
        if self.speed < 0:
            self.speed = 0

    def turn(self, deg):
        """Change the current rotation of the car"""
        if np.absolute(deg) > self.max_turn:
            print(deg)
            assert False
        self.alpha += deg
        self.alpha = self.alpha % 360

    def get_speed_coor(self):
        cst: float = np.pi / 180
        dx: float = self.speed * np.cos(self.alpha * cst)
        dy: float = self.speed * np.sin(self.alpha * cst)
        return Coor((dx,dy))

    def move(self):
        """Change the coordinate of the care according to its speed and alpha"""
        speed_increase = 0
        if self.previous_speed < self.speed:
            speed_increase = 1
        elif self.previous_speed > self.speed:
            speed_increase = -1
        self.previous_speed = self.speed

        dx,dy = self.get_speed_coor().get()
        self.coor.x += dx
        self.coor.y += dy
        self.trajectory.append([Coor((self.coor.x, self.coor.y)), speed_increase])

    def dic(self):
        return {"coor":self.coor, "speed":self.speed, "alpha":self.alpha, "trajectory":self.trajectory}

    def plot(self, markersize=8, vector_constant=2, show_trajectory=False, head_width=1):
        """Plot the car and is speed vectors"""
        # Plot car
        x,y = self.coor.get()
        plt.axis("off")

        # Plot 
        if show_trajectory:
            liste_x = [i[0].x for i in self.trajectory]
            liste_y = [i[0].y for i in self.trajectory]

            for i in range(1, len(self.trajectory)):
                color = "yellow"
                if self.trajectory[i][1] == 1:
                    color = "green"
                elif self.trajectory[i][1] == -1:
                    color = "red"
        
                plt.plot([liste_x[i-1], liste_x[i]], [liste_y[i-1], liste_y[i]], "-o", color=color, markersize=2)
                 
        # Plot car's directoin
        cst: float = np.pi / 180
        dx: float = np.cos(self.alpha * cst)
        dy: float = np.sin(self.alpha * cst)
        plt.arrow(x, y, dx/10, dy/10, head_width=head_width)
        plt.plot([x, x+ dx*self.speed*vector_constant], [y, y+ dy*self.speed*vector_constant], "-", color="red")
        plt.plot(x, y, "o", color='blue', markersize=markersize)


MAX_SPEED = 50
MAX_TURN = 20

class RacingCar(Env):
    def __init__(self):
        super(RacingCar, self).__init__()
        # time between two frames
        self.time = 0.9 #Change this variable to "discretiser" the time. Lower value means more discretisation

        self.max_turn = int(MAX_TURN * self.time)
        self.nb_state = 6*self.max_turn + 3
        self.nb_state = 9
        self.max_speed = int(MAX_SPEED * self.time)

        # Define an action space ranging from 0 to 3
        self.action_space = [self.int_to_action(i) for i in range(self.nb_state)]
        self.int_action_space = [i for i in range(self.nb_state)]

        self.track: Track = None
        self.id_line_goal = 0

        # Define the anle of which we will look the distance
        self.liste_alpha = [60, 40, 20, 0, -20, -40, -60]
        self.max_dist_wall = None

        self.reward_max = 200
        
        self.car: Car = None

    def create_car(self):
        car = Car(self.track.start, self.time)
        return car

    def action_to_int(self, action):
        """Transform an action (tuple) into an action (int)"""
        a,b = action
        return 3*(a+1) + int(b/self.max_turn) +1
        return 3*(b+self.max_turn) + a+1
    
    def int_to_action(self, x):
        """Transform an action (int) into an action (tuple)"""
        return (int(x/3)-1, (x%3 -1)*self.max_turn)
        return ((x%3)-1, int(x/3) -self.max_turn)


    def get_state(self):
        """Return actual state of the env"""
        state = [self.car.speed]
        for alpha in self.liste_alpha:
            coor = self.track.next_wall(self.car.coor, self.car.alpha + alpha, dist_max=self.max_dist_wall)
            state.append(self.car.coor.dist(coor))
        return state

    def reset(self, track):
        """Reset the environment"""
        self.id_line_goal = 0
        self.track = track
        self.max_dist_wall = self.track.height + self.track.width

        self.car = self.create_car()
        return self.get_state(), []

    def render(self, waiting_time=0.01,
               show_trajectory=False, show_dist_to_wall=False,
               show_track_midpoint=False, show_track_lines=False):
        """Render the environment"""
        self.track.plot(hide=True, show_lines=show_track_lines, show_midpoints=show_track_midpoint)
        if show_dist_to_wall:
            for alpha in self.liste_alpha:
                coor = self.track.next_wall(self.car.coor, self.car.alpha + alpha, dist_max=self.max_dist_wall)
                plt.plot([self.car.coor.x, coor.x], [self.car.coor.y, coor.y], "-", color="grey")

        self.car.plot(show_trajectory=show_trajectory)
        display.clear_output(wait=True)
        plt.show()
        time.sleep(waiting_time)
        
    def step(self, action:int):
        """Do a step, we suppose that the action is a possible one"""
        is_done = False
        reward = 0

        x,y = self.car.coor.get()
        previous_coor = Coor((x,y))

        acc, turn = self.int_to_action(action)
        if acc==-1:
            self.car.brake()
        elif acc==1:
            self.car.accelerate()
        self.car.turn(turn)
        self.car.move()

        new_coor = self.car.coor

        has_crashed = False
        if not self.track.is_move_possible(previous_coor, new_coor):
            has_crashed = True
            reward -= 500
            is_done = True

        if self.car.speed == 0:
            reward -= 10
        
        reward += self.car.speed/10
        
        previous_id = (self.id_line_goal-1) % (len(self.track.lines))

        if intersect(self.track.lines[previous_id][0], self.track.lines[previous_id][1], previous_coor, new_coor):
            reward -= 200
            is_done = True

        while intersect(self.track.lines[self.id_line_goal][0], self.track.lines[self.id_line_goal][1], previous_coor, new_coor) and not has_crashed:
            reward += 10
            self.id_line_goal = (self.id_line_goal + 1) % (len(self.track.lines))

        reward -= 1
        return self.get_state(), reward, is_done, has_crashed, []
    
    def random_action(self, p_accel=0.25, p_brake=0.25, p_turn=0.5):
        """Return random possible action according to probability"""
        action = [0,0]
        rd_accel = rd.random()
        if rd_accel <= p_accel:
            action[0] = 1
        elif rd_accel <= p_accel + p_brake:
            action[0] = -1
        
        if rd.random() <= p_turn:
            action[1] = ((-1)**(rd.randint(0,1))) * rd.randint(-self.max_turn, self.max_turn)
        return tuple(action)


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


def compute_data():
    liste_training_time_min = [10, 40, 60]
    liste_training_time_sec = [60*x for x in liste_training_time_min]

    nb_track_training = [8, 40, int(0.8*len(TRACKS))]

    folder_name =  "run_part1_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder_name)
    for nb_track in nb_track_training:
        for time_max in liste_training_time_sec:
            best_genome_path = folder_name + '/best_genome_' + str(time_max) + "sec_" + str(nb_track) + "tracks"
            json_object = json.dumps(training(time_bound=time_max, track_budjet=nb_track, do_save=True, save_path=best_genome_path))
            
            with open(folder_name + "/" + str(time_max) + "_" + str(nb_track) + ".json", "w") as f :
                f.write(json_object)

compute_data()


