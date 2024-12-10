import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random as rd
import time
from os import walk
from gym import Env

from coor import Coor, intersect
from track import Track, create_tracks, save_tracks_lines
from car import Car


TRACKS_FOLDER = "../tracks/png/"
LINES_FOLDER = "../tracks/track_lines/"
PATHS = []
for (dirpath, dirnames, filenames) in walk(TRACKS_FOLDER):
    for file in filenames:
        file_path = TRACKS_FOLDER + file
        PATHS.append((file_path, file[:-4]))
    break

PATHS.sort()
rd.shuffle(PATHS)
print("Number of tracks:", len(PATHS))
print(PATHS)

#save_tracks_lines(PATHS, LINES_FOLDER) # uncomment to recalculate lines
TRACKS:list[Track] = create_tracks(PATHS, LINES_FOLDER)
print(len(TRACKS))


MAX_SPEED = 50
MAX_TURN = 20
class RacingCar(Env):
    def __init__(self):
        super(RacingCar, self).__init__()
        # time between two frames
        # Basic value = 0.9
        self.time = 0.9 #Change this variable to "discretiser" the time. Lower value means more discretisation

        self.max_turn = int(MAX_TURN * self.time)
        self.nb_state = 6*self.max_turn + 3
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
        return 3*(b+self.max_turn) + a+1
    
    def int_to_action(self, x):
        """Transform an action (int) into an action (tuple)"""
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


