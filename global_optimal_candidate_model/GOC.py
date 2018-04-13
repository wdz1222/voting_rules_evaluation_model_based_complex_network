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
        self.epsilon = 10e-4

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
            ip_attr[i, :] = self.caculate_sup_and_cpms(self.intersection_point[i, :])
        return ip_attr

    def caculate_sup_and_cpms(self, point):
        attr = np.zeros(2)
        for j in range(self.n):
            dis = norm(self.voters[j, :] - point)
            if dis <= self.r[j]:
                attr[0] = attr[0] + self.w[j]
                attr[1] = attr[1] + self.w[j] * dis
            else:
                attr[1] = attr[1] + self.w[j] * (dis - self.r[j])
        return attr

    def tri_to_triValue(self):
        '''
        每个节点被转换为四元组（X坐标，y坐标，支持度，妥协度）
        :return:
        '''
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
        return np.array(tri_value)

    @staticmethod
    def delete_invalid_tri(tri_value):
        del_tri_id = list()
        for i in range(len(tri_value)):
            if np.max(tri_value[i][:, 2]) < 0.5:
                del_tri_id.append(i)
        return np.delete(tri_value, del_tri_id, axis=0)

    def caculate_LB_UB(self, tri_value):
        tri_value_len = len(tri_value)
        tri_bounds = np.zeros(tri_value_len, dtype=np.float)
        largest_LB = -1
        print(np.min(tri_value[:][:, 3]))
        for i in range(tri_value_len):
            tri_bounds[i] = np.max(tri_value[i][:, 3])
            lb_loc = np.argmin(tri_value[i][:, 3])
            lb = np.min(tri_value[i][:, 3])
            if tri_value[i][lb_loc, 2] >= 0.5 and lb > largest_LB:
                largest_LB = np.min(tri_value[i][:, 3])
                solution = tri_value[i][lb_loc]
            del_tri_id = list()
        for i in range(len(tri_value)):
            if tri_bounds[i] < largest_LB*(1+self.epsilon):
                del_tri_id.append(i)
        return np.delete(tri_value, del_tri_id, axis=0), solution

    def midpoint(self, x, y):
        point = list()
        loc = (x + y) / 2
        sup = 0.0
        obj_value = 0.0
        for i in range(self.n):
            dis = norm(self.voters[i, :] - loc)
            if dis <= self.r[i]:
                sup = sup + self.w[i]
                obj_value = obj_value + self.w[i] * dis
            else:
                obj_value = obj_value + self.w[i] * (dis - self.r[i])
        point.append(loc[0])
        point.append(loc[1])
        point.append(sup)
        point.append(obj_value)
        return point

    def divide_triangle(self, tri_value):
        triangle = tri_value
        for i in range(len(tri_value)):
            tri = tri_value[i]
            p1 = self.midpoint(tri[0, [0, 1]], tri[1, [0, 1]])
            p2 = self.midpoint(tri[0, [0, 1]], tri[2, [0, 1]])
            p3 = self.midpoint(tri[1, [0, 1]], tri[2, [0, 1]])
            tri1 = np.array([[list(tri[0]), p1, p2]])
            tri2 = np.array([[list(tri[1]), p1, p3]])
            tri3 = np.array([[list(tri[2]), p2, p3]])
            tri4 = np.array([[p1, p2, p3]])
            triangle = np.concatenate((triangle, tri1, tri2, tri3, tri4))
        return triangle

    def caculate_lowest_LB(self, tri_value):
        tri_len = len(tri_value)
        tris_LB = np.zeros([tri_len, 2])
        for i in range(tri_len):
            tris_LB[i, 0] = np.min(tri_value[i][:, 3])
            tris_LB[i, 1] = np.argmin(tri_value[i][:, 3])
        LB = 9999
        UB = -1
        solution = np.zeros(4) - 1
        for i in range(tri_len):
            loc = int(tris_LB[i, 1])
            if tri_value[i][loc, 2] >= 0.5:
                if tri_value[i][loc, 3] <= LB:
                    LB = tri_value[i][loc, 3]
                    solution = tri_value[i][loc].copy()
                elif tri_value[i][loc, 3] >= UB:
                    UB = tri_value[i][loc, 3]
        delete_id = []
        for i in range(tri_len):
            if tris_LB[i, 0] > LB+(UB-LB)/3:
                delete_id.append(i)
        return np.delete(tri_value, delete_id, axis=0), LB, solution

    def caculate_largest_sup(self, tri_value):
        tri_len = len(tri_value)
        tri_sup_bound = np.zeros([tri_len, 2])
        for i in range(tri_len):
            tri_sup_bound[i, 0] = np.max(tri_value[i][:, 2])
            tri_sup_bound[i, 1] = np.argmax(tri_value[i][:, 2])
        preserve_tri_id = np.where(tri_sup_bound[:, 0] == np.max(tri_sup_bound[:, 0]))[0]
        LB = 9999999
        for i in preserve_tri_id:
            # print(tri_value[i][int(tri_sup_bound[i, 1])])
            loc = int(tri_sup_bound[i, 1])
            if tri_value[i][loc, 3] <= LB:
                LB = tri_value[i][loc, 3]
                solution = tri_value[i][loc].copy()
        tri_value = tri_value[preserve_tri_id]
        return tri_value, solution, LB

    def BTST_weber_improved(self):
        tri_value = self.tri_to_triValue()
        tri_value = self.delete_invalid_tri(tri_value)
        if len(tri_value) == 0:
            return np.zeros(4) - 1
        LB_value = 9999999
        itr_num = 0
        while True:
            tri_value, LB_value_current, solution = self.caculate_lowest_LB(tri_value)
            if abs(LB_value - LB_value_current) <= self.epsilon:
                return solution
            LB_value = LB_value_current
            tri_value = self.divide_triangle(tri_value)
            tri_value = self.delete_invalid_tri(tri_value)
            print(len(tri_value))
            itr_num += 1
            if itr_num > 4:
                return solution


    def global_approval_candidate(self):
        tri_value = self.tri_to_triValue()
        tri_value = self.delete_invalid_tri(tri_value)
        if len(tri_value) == 0:
            return np.zeros(4) - 1
        LB_value = 9999999
        while True:
            tri_value, solution, LB_value_current = self.caculate_largest_sup(tri_value)
            # print('LB_value_current=', LB_value_current)
            # print('Largest_sup = ', solution[2])
            if abs(LB_value - LB_value_current) <= self.epsilon:
                return solution
            LB_value = LB_value_current
            tri_value = self.divide_triangle(tri_value)
            tri_value = self.delete_invalid_tri(tri_value)

    def weber(self):
        solution = np.zeros(2)
        while True:
            current_solution = np.zeros(2)
            stemp = 0
            for i in range(self.n):
                dis = norm(solution - self.voters[i])
                current_solution += (self.w[i] * self.voters[i]) / dis
                stemp += self.w[i] / dis
            current_solution = current_solution / stemp
            if norm(current_solution - solution) <= self.epsilon:
                return np.append(current_solution, self.caculate_sup_and_cpms(current_solution))
            solution = current_solution

    def approval_and_condorcet(self):
        attrs = np.zeros([self.m, 2])
        for i in range(self.m):
            attrs[i] = self.caculate_sup_and_cpms(self.candidates[i])
        aw_loc = np.argmax(attrs[:, 0])
        approval_winner = np.append(self.candidates[aw_loc], attrs[aw_loc])
        condorcet_matrix = np.zeros([self.m, self.m])
        for i in range(self.m):
            for j in range(i, self.m):
                if i == j:
                    condorcet_matrix[i, j] = 100
                    continue
                for z in range(self.n):
                    dis1 = norm(self.candidates[i] - self.voters[z])
                    dis2 = norm(self.candidates[j] - self.voters[z])
                    if dis1 < dis2:
                        condorcet_matrix[i, j] += self.w[z]
                    elif dis1 > dis2:
                        condorcet_matrix[j, i] += self.w[z]
                    else:
                        condorcet_matrix[i, j] += self.w[z]
                        condorcet_matrix[j, i] += self.w[z]
        con_loc = np.argmax(np.min(condorcet_matrix, axis=1))
        condorcet_winner = np.append(self.candidates[con_loc], attrs[con_loc])
        return approval_winner, condorcet_winner


def corr_of_voter_num_and_validity_in_weber(voter_number, pdmin, pdmax):
    iter_num = 200
    file_name = 'table1_weber_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    with open('data/table1/'+file_name, 'a') as f:
        for i in range(iter_num):
            goc = GOC(voter_number, 3, pdmin, pdmax)
            weber_solution = goc.weber()
            print('weber_solution = ', weber_solution)
            f.writelines([str(weber_solution[0]), ' ', str(weber_solution[1]), ' ', str(weber_solution[2]), ' ',
                          str(weber_solution[3]), '\n'])


def corr_of_voter_num_and_validity_in_improved_weber(voter_number, pdmin, pdmax):
    iter_num = 200
    file_name = 'table1_improved_weber_'+str(voter_number)+'_'+str(pdmin)+'_'+str(pdmax)+'.txt'
    with open('data/table1/'+file_name, 'a') as f:
        for i in range(iter_num):
            print(i)
            goc = GOC(voter_number, 3, pdmin, pdmax)
            goc.delaunay()
            solution = goc.BTST_weber_improved()
            print('solution = ', solution)
            f.writelines([str(solution[0]), ' ', str(solution[1]), ' ', str(solution[2]), ' ',
                          str(solution[3]), '\n'])


# corr_of_voter_num_and_validity_in_weber(500, 0, 1)
corr_of_voter_num_and_validity_in_improved_weber(40, 0.1, 0.4)

# goc = GOC(20, 3, 0, 0.5)
# approval_winner, condorcet_winner = goc.approval_and_condorcet()
# print('approval_winner = ', approval_winner)
# print('condorcet_winner = ', condorcet_winner)
# weber_solution = goc.weber()
# print('weber_solution = ', weber_solution)
# goc.delaunay()
# weber_improved_solution = goc.BTST_weber_improved()
# print('weber_improved_solution = ', weber_improved_solution)
# ga_solution = goc.global_approval_candidate()
# print('ga_solution = ', ga_solution)
