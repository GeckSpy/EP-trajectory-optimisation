import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

from coor import Coor, intersect

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

def save_lines(track:Track, path:str):
    file = open(path, "w")
    for c1,c2 in track.lines:
        txt = str(c1.x) + "," + str(c1.y) + "," + str(c2.x) + "," + str(c2.y) +"\n" 
        file.write(txt)
    file.close()


def save_tracks_lines(PATHS, LINES_FOLDER):
    # put all the tracks path here:
    # The track called O3.png is not working because of unsmooth corner
    for name, number in PATHS:
        track_info = create_track_info(name)
        track = Track(track_info, nb_lines=2, l_bt_lines=8)
        save_lines(track, LINES_FOLDER + number + ".txt")
        print(name, number)


def create_tracks(PATHS, LINES_FOLDER, show=False) -> list[Track]:
    tracks = []
    counter = 0
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

        if show:
            tracks[-1].plot(show_lines=True, show_midpoints=True)
        print(counter, name, "as", len(tracks[-1].lines), "lines")
        counter +=1

    return tracks

track = create_tracks([["tracks/png/42.png", "42"]], "tracks/track_lines/")[0]
track.plot()