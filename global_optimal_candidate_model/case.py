from GOC import *
import numpy as np
from numpy.linalg import norm


def replacement(d0, d1, z0, z1):
    return (d0 - d1) / (z1 - z0)


def caculate_zw(point, voters, w):
    result = 0
    for i in range(len(voters)):
        result = result + w[i] * norm(point - voters[i])
    return result


for itr in range(300):
    print(itr)
    goc = GOC(30, 5, 0.1, 0.3)
    weber_solution = goc.weber()
    goc.delaunay()
    tri_value = goc.tri_to_triValue()
    weber_improved_solution = goc.BTST_weber_improved(tri_value.copy())
    iga_solution = goc.improved_global_approval_candidate(tri_value.copy())
    approval_winner, condorcet_winner = goc.approval_and_condorcet()
    r = replacement(weber_improved_solution[4], iga_solution[4], weber_improved_solution[3], iga_solution[3])
    if approval_winner[0] != condorcet_winner[0] and approval_winner[1] != condorcet_winner[1]:
        print('approval_winner = ', approval_winner)
        print('condorcet_winner = ', condorcet_winner)
        print('weber_solution = ', weber_solution)
        print('weber_improved_solution = ', weber_improved_solution)
        print('iga_solution = ', iga_solution)
        print('replacement = ', r)
        zw1 = caculate_zw(condorcet_winner[[0, 1]], goc.voters, goc.w)
        zw2 = caculate_zw(approval_winner[[0, 1]], goc.voters, goc.w)
        zw3 = caculate_zw(weber_solution[[0, 1]], goc.voters, goc.w)
        zw4 = caculate_zw(weber_improved_solution[[0, 1]], goc.voters, goc.w)
        zw5 = caculate_zw(iga_solution[[0, 1]], goc.voters, goc.w)
        re1 = zw1/zw3
        re2 = zw2/zw3
        if r < 0 and (zw1-zw2)*(condorcet_winner[3]-approval_winner[3]) < 0 and re1 <= 3 and re2 <= 3:
            s = np.empty([5, 5])
            s1 = np.array([condorcet_winner, approval_winner, weber_solution, weber_improved_solution, iga_solution])
            s[:, [0, 1]] = s1[:, [0, 1]]
            s[:, 2] = s1[:, 4]
            s[:, 3] = s1[:, 3]
            s[0, 4] = zw1
            s[1, 4] = zw2
            s[2, 4] = zw3
            s[3, 4] = zw4
            s[4, 4] = zw5
            s = s.T
            with open('data/table5/c2', 'a') as f1:
                f1.writelines(str(r) + '\n')
                for t in range(5):
                    f1.writelines(' '.join([str(v) for v in s[t, :]]) + '\n')
                f1.writelines('\n' + '---------------------------------------------' + '\n')