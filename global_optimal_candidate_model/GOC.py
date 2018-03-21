import numpy as np
from numpy.linalg import norm
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
        self.intersection_point = 0
        self.tri = 0

    def create_w(self):
        '''
        给每个节点赋予权值，并且权值和为1
        :return: w:权值向量
        '''
        w = np.random.rand(self.n)
        si = w[0]
        for i in range(self.n-2):
            w[i+1] = np.random.rand()*(1-si)
            si += w[i+1]
        w[-1] = 1-si
        return w

    def crt_intersection(self):
        '''
        计算节点偏好域之间的交点
        :return: intersection_point:所有节点偏好域之间的节点集合
        '''
        intersection_point = []
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                l = norm(self.voters[i, :]-self.voters[j, :])
                if abs(self.r[i]-self.r[j]) < l < self.r[i]+self.r[j]:
                    points = self.points_of_intersection(i, j, l)
                    for z in range(len(points)):
                        if 0 < points[z][0] < 1 and 0 < points[z][1] < 1:
                            intersection_point.append(points[z])
                elif l <= abs(self.r[i]-self.r[j]):
                    intersection_point.append(self.voters[i, :])
                    intersection_point.append(self.voters[j, :])
        return np.array(intersection_point)

    def points_of_intersection(self, i, j, l):
        '''
        计算两个节点偏好域之间的交点
        :param i: 节点i
        :param j: 节点j
        :param l: 两个节点之间的偏好距离
        :return: 偏好域交点
        '''
        ael = (np.power(self.r[i], 2)-np.power(self.r[j], 2)+np.power(l, 2))/(2*np.power(l, 2))
        x0 = self.voters[i, 0]+(self.voters[j, 0]-self.voters[i, 0])*ael
        k1 = (self.voters[j, 1]-self.voters[i, 1])/(self.voters[j, 0]-self.voters[i, 0])
        k2 = -1/k1
        y0 = self.voters[i, 1]+(self.voters[j, 1]-self.voters[i, 1])*ael
        r2 = np.power(self.r[i], 2)-np.power(x0-self.voters[i, 0], 2)-np.power(y0-self.voters[i, 1], 2)
        ef = np.sqrt(r2)/np.sqrt(1+np.power(k2, 2))
        xc = x0-ef
        yc = y0+k2*(xc-x0)
        xd = x0+ef
        yd = y0+k2*(xd-x0)
        # print([[xc, yc], [xd, yd]])
        return [[xc, yc], [xd, yd]]

    def delaunay(self):
        '''
        根据返回的偏好域交点集合构建三角剖分划分解空间
        :return:
        tri: 返回三角剖分节点集合
        interaction_point: 偏好交叉节点（即候选解）
        '''
        self.intersection_point = self.crt_intersection()
        tri = Delaunay(self.intersection_point)
        self.tri = tri.simplices.copy()
        # print(tri.simplices)
        # plt.triplot(self.intersection_point[:, 0], self.intersection_point[:, 1], self.tri)
        # plt.plot(self.intersection_point[:, 0], self.intersection_point[:, 1], 'o')
        # plt.show()

    def caculate_ip_attr(self):
        # 初始化可行解参数
        len_ip = len(self.intersection_point)
        ip_attr = np.zeros([len_ip, 2], dtype=np.float)
        for i in range(len_ip):
            for j in range(self.n):
                dis = norm(self.voters[j, :] - self.intersection_point[i, :])
                if dis <= self.r[j]:
                    ip_attr[i, 0] = ip_attr[i, 0] + self.w[j]
                    ip_attr[i, 1] = ip_attr[i, 1] + self.w[j]*dis
                else:
                    ip_attr[i, 1] = ip_attr[i, 1] + self.w[j]*(dis-self.r[j])
        return ip_attr

    def tri_to_triValue(self):
        ip_attr = self.caculate_ip_attr()
        tri_value = list()
        for i in range(len(self.tri)):
            tri_value_i = list()
            for j in range(3):
                v = list()
                v.append(self.intersection_point[self.tri[i, j], 0])
                v.append(self.intersection_point[self.tri[i, j], 1])
                v.append(ip_attr[self.tri[i, j], 0])
                v.append(ip_attr[self.tri[i, j], 1])
                tri_value_i.append(v)
            tri_value.append(tri_value_i)
        return tri_value

    @staticmethod
    def delete_invalid_tri(tri_value):
        del_tri_id = list()
        print(tri_value[0][:, 1])
        for i in range(len(tri_value)):
            if np.max(tri_value[i][:, 2]) < 0.5:
                del_tri_id.append(i)
        np.delete(tri_value, del_tri_id, axis=0)
        return tri_value

    def BTST_weber_improved(self):
        tri_value = self.tri_to_triValue()
        print(len(tri_value))
        tri_value = self.delete_invalid_tri(tri_value)
        print(tri_value)
        # if np.max(ip_attr[:, 0]) < 0.5:
        #     print('no solution')
        #     return -1
        # tri = self.delete_invalid_ip(tri, ip_attr)
        # tri_len = len(tri)
        # while tri_len > 0:



goc = GOC(10, 3, 0, 0.5)
goc.delaunay()
goc.BTST_weber_improved()
