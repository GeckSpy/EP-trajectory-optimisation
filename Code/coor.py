import numpy as np

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