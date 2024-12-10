import matplotlib.pyplot as plt
import numpy as np
from coor import Coor

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

