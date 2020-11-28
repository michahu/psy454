from itertools import permutations
import glob, os
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

def get_data(file):
    # create nC2 combinations of lines in a file, split into data and target
    data = []
    target = []

    with open(file) as f:
        lines = f.read().splitlines()
        # print(lines)
        perms = permutations(lines, 2)
        for line in perms:
            analogy1, analogy2 = line
            analogy1 = analogy1.split()
            analogy2 = analogy2.split()
            # print(analogy1, analogy2)
            dat = analogy1[0] + " " + analogy1[1] + " " + analogy2[0]
            targ = analogy2[1] 
            data.append(dat)
            target.append(targ)
    
    return list(zip(data, target))

# print(get_data('./Testing/Phase1Answers-1a.txt'))

# for filename in glob.glob('./Testing/*.txt'):
# all_data = np.empty((0, 2))
# path = str(os.getcwd()) + '/Testing/'
# for filename in os.listdir(path):
#     print(filename)
#     # with open(path+filename, 'r') as f:
#     # all_data.append(get_data(path+filename))
#     all_data = np.vstack((all_data, get_data(path+filename)))

# print(all_data.shape)
# df = pd.DataFrame(all_data, columns=['data', 'target'])
# df.to_csv('test-analogies.csv', index=False)

# df = pd.read_csv('train-analogies.csv')
# train, val = train_test_split(df, test_size = 0.2)
# train.to_csv('train-analogies-80.csv', index=False)
# val.to_csv('val-analogies.csv', index=False)            

path = str(os.getcwd()) + '/Meta-Training/'
count = 0
for filename in os.listdir(path):
    os.rename(path+filename, str(count) + '.csv')
    count += 1