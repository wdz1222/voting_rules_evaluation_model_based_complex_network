import numpy as np
from sympy import *
from numpy.linalg import norm
from sympy.geometry import Circle
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class GOC:

    def __init__(self, n, m, pdmin, pdmax):
        self.n = n
        self.m = m
        self.pdmin = pdmin
        self.pdmax = pdmax
        self.voters = np.random.rand(self.n, 2)
        self.candidates = np.random.rand(self.m, 2)
        self.r = np.random.rand(self.n)*(self.pdmax-self.pdmin)+self.pdmin
        self.w = self.create_w()

    def create_w(self):
        w = np.random.rand(self.n)
        si = w[0]
        for i in range(self.n-2):
            w[i+1] = np.random.rand()*(1-si)
            si += w[i+1]
        w[-1] = 1-si
        return w

    def crt_intersection(self):
        intersection_point = []
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                cd = norm(self.voters[i, :]-self.voters[j, :])
                if abs(self.r[i]-self.r[j]) < cd < self.r[i]+self.r[j]:
                    c1 = Circle(Point(self.voters[i, :]), self.r[i])
                    c2 = Circle(Point(self.voters[j, :]), self.r[j])
                    points = c1.intersection(c2)
                    for z in range(len(points)):
                        if 0 < float(points[z].x) < 1 and 0 < float(points[z].y) < 1:
                            intersection_point.append([float(points[z].x), float(points[z].y)])
                elif cd <= abs(self.r[i]-self.r[j]):
                    intersection_point.append(self.voters[i, :])
                    intersection_point.append(self.voters[j, :])
        return np.array(intersection_point)

    def delaunay(self):
        interaction_point = self.crt_intersection()
        print(interaction_point)
        tri = Delaunay(interaction_point)
        plt.triplot(interaction_point[:, 0], interaction_point[:, 1], tri.simplices.copy())
        plt.plot(interaction_point[:, 0], interaction_point[:, 1], 'o')
        plt.show()


goc = GOC(100, 3, 0, 1)
goc.delaunay()
