import matplotlib.pyplot as plt
from IPython import display
from gym import Env, spaces
import random as rd
import numpy as np
import time


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
        return (self.x==coor2.x) and (self.y==coor2.y)
    
    def dist(self, coor2):
        a,b = self.get()
        c,d = self.get()
        return np.sqrt(a*c + b*d)


N = 5
n = 5

RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
GREY = [70 for _ in range(3)]
WHITE = [255 for _ in range(3)]

START_CHAR = "s"
END_CHAR = "e"

CAR_ICON = [[BLUE]]

def color(b):
    if b == START_CHAR:
        return GREEN
    elif b == END_CHAR:
        return RED
    elif b == 1:
        return GREY
    else:
        return WHITE


class Track():
    def __init__(self, tab):
        self.height = len(tab)
        self.width = len((tab[0]))
        self.info_track = tab
        self.color_track = [[color(x) for x in y] for y in tab]

        self.end = None
        self.start = None
        for i in range(self.height):
            for j in range(self.width):
                if tab[i][j] == START_CHAR:
                    self.start = Coor((i,j))
                if tab[i][j] == END_CHAR:
                    self.end = Coor((i,j))

    def get_color(self, coor:Coor):
        """return the color of the case x,y"""
        x,y = coor.get()
        return color(self.info_track[x][y])
    
    def is_wall(self, coor:Coor):
        """Return True if case (x,y) is a wall"""
        x,y = coor.get()
        return (self.info_track[x][y] == 1)

    def get_start(self):
        """Return coordinate of start"""
        return self.start.get()
    
    def get_end(self):
        """Return coordinate of end"""
        return self.end.get()

    def plot(self):
        """Plot the track using matplotlib"""
        plt.imshow(self.color_track)
        plt.axis("off")
        plt.show()


track_1_bool = [[1, 1, 1, 1, "e"],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                ["s", 0, 0, 0, 1]]

track_1 = Track(track_1_bool)
print("start at coordinates:", track_1.get_start())


class MyEnv(Env):
    def __init__(self, track: Track):
        super(MyEnv, self).__init__()

        # Define a 2-D observation space
        self.observation_shape = (N, N, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.int64)

        # Define an action space ranging from 0 to 3
        self.action_space = [i for i in range(4)]

        self.track = track
        self.car = Coor(self.track.get_start())
        # self.car_icon = cv2.imread("blue_dot.png") /255
        self.car_icon = np.array(CAR_ICON)

        self.canvas = np.array(self.track.color_track)

        # self.action_meanings = {0: "Right", 1: "Left", 2: "Down", 3: "Up"}


    def draw_car(self):
        """Add the car_icon to the canvas"""
        car_shape = self.car_icon.shape
        x,y = self.car.get()
        self.canvas[x:x + car_shape[1], y:y + car_shape[0]] = self.car_icon

    def reset(self):
        """Reset the environment"""
        self.car = Coor(self.track.get_start())
        self.canvas = np.array(self.track.color_track)
        self.draw_car()
        return self.canvas

    def render(self, mode = "human", waiting_time=0.1):
        """Render the environment"""
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        self.canvas = np.array(self.track.color_track)
        self.draw_car()
        if mode == "human":
            plt.imshow(self.canvas)
            plt.axis("off")
            display.clear_output(wait=True)
            plt.show()
            time.sleep(waiting_time)
    
        elif mode == "rgb_array":
            return self.canvas
        

    def move(self, action):
        """Return the move cooresponding to action"""
        if action == 0:
            return Coor((0, 1))
        elif action == 1:
            return Coor((0, -1))
        elif action == 2:
            return Coor((1, 0)) #because top is at row 0
        else:
            return Coor((-1,0))

    def is_case_ridable(self, coor: Coor):
        """Return if the car can go on the case or not"""
        x,y = coor.get()
        if not (x>=0 and x<self.track.height and y>=0 and y<self.track.width):
            return False
        return not self.track.is_wall(coor)
    
    def possible_action(self):
        actions = []
        if self.is_case_ridable(self.car + Coor((0,-1))):
            actions.append(1)
        if self.is_case_ridable(self.car + Coor((0, 1))):
            actions.append(0)
        if self.is_case_ridable(self.car + Coor((1, 0))):
            actions.append(2)
        if self.is_case_ridable(self.car + Coor((-1,0))):
            actions.append(3)
        return actions
        
    def step(self, action):
        """Do a step, we suppose that the action is a possible one"""
        is_done = False
        reward = -1

        mv = self.move(action)
        next_move = self.car + mv
        reward += next_move.dist(self.track.end) - self.car.dist(self.track.end)
        self.car = next_move

        if self.car == self.track.end:
            reward = 200
            is_done = True

        return self.canvas, reward, is_done, []
        
        

env = MyEnv(track_1)
obs = env.reset()


import random as rd

running = False
step = 0
while running:
    step += 1
    # Take a random action
    actions = env.possible_action()
    action = rd.choice(actions)
    obs, reward, done, info = env.step(action)

    # Render the game
    env.render(waiting_time=0.01)

    if done == True or step > 1000:
        running = False

print("number of steps:", step)

env.close()