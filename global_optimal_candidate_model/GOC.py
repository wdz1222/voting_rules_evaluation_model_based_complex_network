#coding=utf-8
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
        # self.r = np.array([1.4]*self.n)
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
                        intersection_point.append(points[z])
                        # if 0 < points[z][0] < 1 and 0 < points[z][1] < 1:
                        #     intersection_point.append(points[z])
                intersection_point.append(self.voters[i, :])
                intersection_point.append(self.voters[j, :])
                # elif l <= abs(self.r[i]-self.r[j]):
                #     intersection_point.append(self.voters[i, :])
                #     intersection_point.append(self.voters[j, :])
            intersection_point.extend(np.random.rand(30, 2))
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
        ip_attr = np.zeros([len_ip, 3], dtype=np.float)
        for i in range(len_ip):
            ip_attr[i, :] = self.caculate_sup_and_cpms(self.intersection_point[i, :])
        return ip_attr

    def caculate_sup_and_cpms(self, point):
        attr = np.zeros(3)
        for j in range(self.n):
            dis = norm(self.voters[j, :] - point)
            if dis <= self.r[j]:
                attr[0] = attr[0] + self.w[j]
                attr[1] = attr[1] + self.w[j] * dis
                attr[2] = attr[2] + self.w[j] * self.r[j]
            else:
                attr[1] = attr[1] + self.w[j] * (dis - self.r[j])
        return attr

    def tri_to_triValue(self):
        '''
        每个交叉节点节点被转换为四元组（X坐标，y坐标，支持度，妥协度, 加权支持度）
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
                v.append(ip_attr[self.tri[i, j], 2])
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
        wsup = 0.0
        for i in range(self.n):
            dis = norm(self.voters[i, :] - loc)
            if dis <= self.r[i]:
                sup = sup + self.w[i]
                obj_value = obj_value + self.w[i] * dis
                wsup = wsup + self.w[i] * self.r[i]
            else:
                obj_value = obj_value + self.w[i] * (dis - self.r[i])
        point.append(loc[0])
        point.append(loc[1])
        point.append(sup)
        point.append(obj_value)
        point.append(wsup)
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
        LB = 9999
        LB_loc = 0
        solution = np.zeros(4) - 1
        for i in range(tri_len):
            for j in range(3):
                if tri_value[i][j, 2] >= 0.5:
                    if tri_value[i][j, 3] <= LB:
                        LB = tri_value[i][j, 3]
                        solution = tri_value[i][j].copy()
                        LB_loc = i
        return np.array([tri_value[LB_loc]]), LB, solution

    def caculate_largest_sup(self, tri_value):
        tri_len = len(tri_value)
        tri_sup_bound = np.zeros([tri_len, 2])
        maxsup = 0
        for i in range(tri_len):
            tri_sup_bound[i, 0] = np.max(tri_value[i][:, 2])
            if tri_sup_bound[i, 0] >= maxsup:
                maxsup = tri_sup_bound[i, 0]
            tri_sup_bound[i, 1] = np.argmax(tri_value[i][:, 2])
        preserve_tri_id = np.where(tri_sup_bound[:, 0] == maxsup)[0]
        LB = 9999999
        for i in preserve_tri_id:
            # print(tri_value[i][int(tri_sup_bound[i, 1])])
            loc = int(tri_sup_bound[i, 1])
            if tri_value[i][loc, 3] <= LB:
                LB = tri_value[i][loc, 3]
                solution = tri_value[i][loc].copy()
        tri_value = tri_value[preserve_tri_id]
        # print(len(tri_value))
        return tri_value, solution, LB

    def caculate_largest_wsup(self, tri_value):
        tri_len = len(tri_value)
        tri_sup_bound = np.zeros([tri_len, 2])
        maxwsup = 0
        for i in range(tri_len):
            tri_sup_bound[i, 0] = np.max(tri_value[i][:, 4])
            if tri_sup_bound[i, 0] >= maxwsup:
                maxwsup = tri_sup_bound[i, 0]
            tri_sup_bound[i, 1] = np.argmax(tri_value[i][:, 4])
        preserve_tri_id = np.where(tri_sup_bound[:, 0] == maxwsup)[0]
        LB = 9999999
        for i in preserve_tri_id:
            # print(tri_value[i][int(tri_sup_bound[i, 1])])
            loc = int(tri_sup_bound[i, 1])
            if tri_value[i][loc, 3] <= LB:
                LB = tri_value[i][loc, 3]
                solution = tri_value[i][loc].copy()
        tri_value = tri_value[preserve_tri_id]
        # print(len(tri_value))
        return tri_value, solution, LB

    def BTST_weber_improved(self, tri_value):
        tri_value = self.delete_invalid_tri(tri_value)
        if len(tri_value) == 0:
            return np.zeros(5) - 1
        LB_value = 9999999
        itr_num = 0
        while True:
            tri_value, LB_value_current, solution = self.caculate_lowest_LB(tri_value)
            if norm(LB_value - LB_value_current) <= self.epsilon:
                return solution
            LB_value = LB_value_current
            tri_value = self.divide_triangle(tri_value)
            tri_value = self.delete_invalid_tri(tri_value)

    def global_approval_candidate(self, tri_value):
        tri_value = self.delete_invalid_tri(tri_value)
        if len(tri_value) == 0:
            return np.zeros(5) - 1
        LB_value = 9999999
        while True:
            tri_value, solution, LB_value_current = self.caculate_largest_sup(tri_value)
            # print('LB_value_current=', LB_value_current)
            # print('Largest_sup = ', solution[2])
            if norm(LB_value - LB_value_current) <= self.epsilon:
                return solution
            LB_value = LB_value_current
            tri_value = self.divide_triangle(tri_value)
            tri_value = self.delete_invalid_tri(tri_value)

    def improved_global_approval_candidate(self, tri_value):
        tri_value = self.delete_invalid_tri(tri_value)
        if len(tri_value) == 0:
            return np.zeros(5) - 1
        LB_value = 9999999
        while True:
            tri_value, solution, LB_value_current = self.caculate_largest_wsup(tri_value)
            # print('LB_value_current=', LB_value_current)
            # print('Largest_sup = ', solution[2])
            if norm(LB_value - LB_value_current) <= self.epsilon:
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
        attrs = np.zeros([self.m, 3])
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

    def paint1(self, r):
        self.r = np.array([r] * self.n)
        domain = [10, 25]
        minw = np.min(self.w)
        maxw = np.max(self.w)
        k = (domain[1] - domain[0]) / (maxw - minw)
        norv = domain[0] + k * (self.w - minw)
        weber_solution = self.weber()
        self.delaunay()
        tri_value = self.tri_to_triValue()
        weber_improved_solution = self.BTST_weber_improved(tri_value.copy())
        iga_solution = self.improved_global_approval_candidate(tri_value.copy())
        plt.figure(figsize=(8, 8))
        for i in range(self.n):
            plt.plot(self.voters[i, 0], self.voters[i, 1], 'bo', markersize=norv[i])
        plt.plot(weber_solution[0], weber_solution[1], 'rs', markersize=8)
        plt.annotate(r'WEBER',
                     xy=(weber_solution[0], weber_solution[1]), xycoords='data',
                     xytext=(+10, +20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.plot(weber_improved_solution[0], weber_improved_solution[1], 'rs', markersize=8)
        plt.annotate(r'EW',
                     xy=(weber_improved_solution[0], weber_improved_solution[1]), xycoords='data',
                     xytext=(-10, -20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.plot(iga_solution[0], iga_solution[1], 'rs', markersize=8)
        plt.annotate(r'MW',
                     xy=(iga_solution[0], iga_solution[1]), xycoords='data',
                     xytext=(-10, -20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.savefig('data/fig/' + str(r)+'.jpg', format='jpg')
        plt.show()
        plt.close()

def caculate_RE(c, a, w, iw, ga):
    return np.array([c/w, c/iw, c/ga, a/w, a/iw, a/ga])


def iga_experiment(voter_number, pdmin, pdmax):
    iter_num = 200
    file_name3 = 'improved_global_approval_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    for z in range(iter_num):
        goc = GOC(voter_number, 3, pdmin, pdmax)
        goc.delaunay()
        iga_solution = goc.improved_global_approval_candidate()
        print(z)
        print('iga_solution = ', iga_solution)
        with open('data/table1/' + file_name3, 'a') as f:
            f.writelines([str(iga_solution[0]), ' ', str(iga_solution[1]), ' ', str(iga_solution[2]), ' ',
                           str(iga_solution[3]), str(iga_solution[4]), '\n'])
    print('---------------------------------------------------------')


def validity_experiment(voter_number, pdmin, pdmax):
    iter_num = 200
    file_name1 = 'weber_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    file_name2 = 'improved_weber_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    file_name3 = 'global_approval_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    # file_name4 = 'RE_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    file_name5 = 'approval_winner_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    file_name6 = 'condorcet_winner_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    file_name7 = 'improved_global_approval_' + str(voter_number) + '_' + str(pdmin) + '_' + str(pdmax) + '.txt'
    for z in range(iter_num):
        goc = GOC(voter_number, 3, pdmin, pdmax)
        weber_solution = goc.weber()
        goc.delaunay()
        tri_value = goc.tri_to_triValue()
        weber_improved_solution = goc.BTST_weber_improved(tri_value.copy())
        global_approval_solution = goc.global_approval_candidate(tri_value.copy())
        iga_solution = goc.improved_global_approval_candidate(tri_value.copy())
        approval_winner, condorcet_winner = goc.approval_and_condorcet()
        print(z)
        print('approval_winner = ', approval_winner)
        print('condorcet_winner = ', condorcet_winner)
        print('weber_solution = ', weber_solution)
        print('weber_improved_solution = ', weber_improved_solution)
        print('ga_solution = ', global_approval_solution)
        print('iga_solution = ', iga_solution)
        print('-------------------------------------------------------------------------------')
        with open('data/table1/' + file_name1, 'a') as f1:
            f1.writelines([str(weber_solution[0]), ' ', str(weber_solution[1]), ' ', str(weber_solution[2]), ' ',
                          str(weber_solution[3]), ' ', str(weber_solution[4]), '\n'])
        with open('data/table1/' + file_name2, 'a') as f2:
            f2.writelines([str(weber_improved_solution[0]), ' ', str(weber_improved_solution[1]), ' ', str(weber_improved_solution[2]), ' ',
                           str(weber_improved_solution[3]), ' ', str(weber_improved_solution[4]), '\n'])
        with open('data/table1/' + file_name3, 'a') as f3:
            f3.writelines([str(global_approval_solution[0]), ' ', str(global_approval_solution[1]), ' ', str(global_approval_solution[2]), ' ',
                           str(global_approval_solution[3]), ' ', str(global_approval_solution[4]), '\n'])
        # RE = caculate_RE(condorcet_winner[3], approval_winner[3], weber_solution[3], weber_improved_solution[3], global_approval_solution[3])
        # with open('data/table/' + file_name4, 'a') as f4:
        #     f4.writelines([str(RE[0]), ' ', str(RE[1]), ' ', str(RE[2]), ' ', str(RE[3]), ' ', str(RE[4]), ' ', str(RE[5]) + '\n'])
        with open('data/table1/' + file_name5, 'a') as f5:
            f5.writelines([str(approval_winner[0]), ' ', str(approval_winner[1]), ' ', str(approval_winner[2]), ' ',
                           str(approval_winner[3]), ' ', str(approval_winner[4]), '\n'])
        with open('data/table1/' + file_name6, 'a') as f6:
            f6.writelines([str(condorcet_winner[0]), ' ', str(condorcet_winner[1]), ' ', str(condorcet_winner[2]), ' ',
                           str(condorcet_winner[3]), ' ', str(condorcet_winner[3]), '\n'])
        with open('data/table1/' + file_name7, 'a') as f7:
            f7.writelines([str(iga_solution[0]), ' ', str(iga_solution[1]), ' ', str(iga_solution[2]), ' ',
                           str(iga_solution[3]), ' ', str(iga_solution[4]), '\n'])


'''
验证有效性
'''
def table1(model_name):
    num = [20, 40, 100, 200, 500]
    sup_domain = [[0.1, 0.4], [0.7, 1], [0, 1]]
    for n in num:
        print('voter_number = ', n)
        for s in sup_domain:
            file_name = '_'.join([model_name, str(n), str(s[0]), str(s[1])])
            file_path = 'data/table/' + file_name + '.txt'
            data = np.loadtxt(file_path)
            validity = len(np.where(data[:, 2] >= 0.5)[0]) / 200.0
            print('[%f, %f]: %f' % (s[0], s[1], validity))


'''
统计最大最小平均值
'''
def table2(model_name):
    num = [20, 40, 100, 200, 500]
    sup_domain = [[0.1, 0.4], [0.7, 1], [0, 1]]
    for n in num:
        print('voter_number = ', n)
        for s in sup_domain:
            file_name = '_'.join([model_name, str(n), str(s[0]), str(s[1])])
            file_path = 'data/table/' + file_name + '.txt'
            data = np.loadtxt(file_path)
            data = np.delete(data, np.where(data[:, 0] == -1)[0], axis=0)
            supmin = np.min(data[:, 2])
            supmax = np.max(data[:, 2])
            supave = np.average(data[:, 2])
            supvar = np.var(data[:, 2])
            commin = np.min(data[:, 3])
            commax = np.max(data[:, 3])
            comave = np.average(data[:, 3])
            comvar = np.var(data[:, 3])
            print('%f, %f, %f, %f, %f, %f,  %f, %f' % (supmin, supmax,
                                                                 supave, supvar, commin, commax, comave, comvar))


'''
绘图
'''
def paint():
    domain = [10, 25]
    goc = GOC(20, 3, 0, 1)
    minw = np.min(goc.w)
    maxw = np.max(goc.w)
    k = (domain[1] - domain[0]) / (maxw - minw)
    norv = domain[0] + k * (goc.w - minw)
    weber_solution = goc.weber()
    goc.delaunay()
    weber_improved_solution = goc.BTST_weber_improved()
    global_approval_solution = goc.global_approval_candidate()
    approval_winner, condorcet_winner = goc.approval_and_condorcet()
    plt.figure(1)
    for i in range(goc.n):
        plt.plot(goc.voters[i, 0], goc.voters[i, 1], 'bo', markersize=norv[i])
    plt.plot(weber_solution[0], weber_solution[1], 'rs', markersize=8)
    plt.annotate(r'weber',
                 xy=(weber_solution[0], weber_solution[1]), xycoords='data',
                 xytext=(+10, +20), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot(weber_improved_solution[0], weber_improved_solution[1], 'rs', markersize=8)
    plt.annotate(r'improved weber',
                 xy=(weber_improved_solution[0], weber_improved_solution[1]), xycoords='data',
                 xytext=(-10, -20), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot(global_approval_solution[0], global_approval_solution[1], 'rs', markersize=8)
    plt.annotate(r'Global approval',
                 xy=(global_approval_solution[0], global_approval_solution[1]), xycoords='data',
                 xytext=(-10, -20), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    if approval_winner[0]==condorcet_winner[0] and approval_winner[1]==condorcet_winner[1]:
        plt.plot(approval_winner[0], approval_winner[1], 'rs', markersize=8)
        plt.annotate(r'Con&MA',
                     xy=(approval_winner[0], approval_winner[1]), xycoords='data',
                     xytext=(-10, -20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    else:
        plt.plot(approval_winner[0], approval_winner[1], 'rs', markersize=8)
        plt.annotate(r'MA',
                     xy=(approval_winner[0], approval_winner[1]), xycoords='data',
                     xytext=(+10, +20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.plot(condorcet_winner[0], condorcet_winner[1], 'rs', markersize=8)
        plt.annotate(r'Con',
                     xy=(condorcet_winner[0], condorcet_winner[1]), xycoords='data',
                     xytext=(-10, -20), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    for i in range(3):
        if goc.candidates[i, 0] != approval_winner[0] and goc.candidates[i, 0] != condorcet_winner[0]:
            plt.plot(goc.candidates[i, 0], goc.candidates[i, 1], 'rs', markersize=8)
            plt.annotate(r'Loster',
                         xy=(goc.candidates[i, 0], goc.candidates[i, 1]), xycoords='data',
                         xytext=(-10, -20), textcoords='offset points', fontsize=10,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()





# paint1()
# num = [20, 40, 100, 200, 500]
# for n in num:
#     validity_experiment(n, 0, 1)
# table1('global_approval')
# paint()
# table2('global_approval')
# corr_of_voter_num_and_validity_in_weber(500, 0, 1)
# num = [20, 40, 100, 200, 500]
# sup_domain = [[0.1, 0.4], [0.7, 1], [0, 1]]
# for z in range(5):
#     for t in range(3):
#         validity_experiment(num[z], sup_domain[t][0], sup_domain[t][1])
#         print('successful')
# corr_of_voter_num_and_validity_in_global_approval(num[0], sup_domain[2][0], sup_domain[2][1])
# for i in range(5):
#     for j in range(3):
#         corr_of_voter_num_and_validity_in_global_approval(num[i], sup_domain[j][0], sup_domain[j][1])
# for i in range(3):
#     corr_of_voter_num_and_validity_in_improved_weber(500, sup_domain[i][0], sup_domain[i][1])

goc = GOC(50, 3, 0, 1)
# approval_winner, condorcet_winner = goc.approval_and_condorcet()
# print('approval_winner = ', approval_winner)
# print('condorcet_winner = ', condorcet_winner)
weber_solution = goc.weber()
print('weber_solution = ', weber_solution)
goc.delaunay()
tri_value = goc.tri_to_triValue()
# print('delaunay successful')
weber_improved_solution = goc.BTST_weber_improved(tri_value)
print('weber_improved_solution = ', weber_improved_solution)
# ga_solution = goc.global_approval_candidate()
# print('ga_solution = ', ga_solution)
# iga_solution = goc.improved_global_approval_candidate()
# print('iga_solution = ', iga_solution)
# validity_experiment(100, 0, 1)

