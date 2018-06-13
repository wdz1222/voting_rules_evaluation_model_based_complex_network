import numpy as np


def clear(model_name):
    num = [20, 40, 100, 200, 500]
    # sup_domain = [[0.1, 0.4], [0.7, 1], [0, 1]]
    for n in num:
        print('voter_number = ', n)
        file_name = '_'.join([model_name, str(n), str(0), str(1)])
        file_path = 'data/table1/' + file_name + '.txt'
        out_path = 'data/temp/' + file_name + '.txt'
        file = open(file_path)
        for line in file:
            loc = line.rfind('.')
            with open(out_path, 'a') as f:
                f.write(line[0: loc-2] + ' ' + line[loc-1:])
        file.close()


def table2(model_name):
    num = [20, 40, 100, 200, 500]
    # sup_domain = [[0.1, 0.4], [0.7, 1], [0, 1]]
    for n in num:
        # print('voter_number = ', n)
        file_name = '_'.join([model_name, str(n), str(0), str(1)])
        file_path = 'data/temp/' + file_name + '.txt'
        data = np.loadtxt(file_path)
        data = np.delete(data, np.where(data[:, 0] == -1)[0], axis=0)
        isupmin = np.min(data[:, 4])
        isupmax = np.max(data[:, 4])
        isupave = np.average(data[:, 4])
        isupvar = np.var(data[:, 4])
        commin = np.min(data[:, 3])
        commax = np.max(data[:, 3])
        comave = np.average(data[:, 3])
        comvar = np.var(data[:, 3])
        print('%f, %f, %f, %f, %f, %f,  %f, %f' % (commin, comave, commax, comvar, isupmin, isupave, isupmax, isupvar))


# table2('improved_global_approval')
# clear('global_approval')
a = np.array([1, 2, 3])
b = np.array([5, 6, 8])
c = np.array([a, b])
print(' '.join([str(i) for i in a]))