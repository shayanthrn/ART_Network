import numpy as np
import matplotlib.pyplot as plt
from patterns import *
import random

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def print_pattern(pattern):
    plt.matshow(pattern)
    plt.show()

def cosin_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def make_noisy(pattern):
    indexes = random.sample(range(len(pattern)), (len(pattern)*5)//100)
    for index in indexes:
        if(pattern[index] == 1):
            pattern[index] = 0
        if(pattern[index] == 0):
            pattern[index] = 1
    return pattern

# draw inputs
# fig = plt.figure(figsize=(8, 8))
# for i in range(1, 21):
#     img = patterns[i-1]
#     fig.add_subplot(5,4,i)
#     plt.imshow(img)
# plt.show()

train_set = []
for i in range(20):
    train_set.append(patterns[i].flatten())

# clusters_test = [[] for i in range(20)]
# for i in range(20):
#     for j in range(2):
#         clusters[i].append(make_noisy(patterns[i].flatten()))

    # fig = plt.figure(figsize=(8, 8))
    # for k in range(1, 6):
    #     img = clusters[i][k-1].reshape(8,8)
    #     fig.add_subplot(1,5,k)
    #     plt.imshow(img)
    # plt.show()

# train_set = flatten(clusters)
vigilance = 0.9
learning_rate = 2


U_weights = (1/65)*np.ones((20,64))
D_weights = np.ones((64,20))

for pattern in train_set:
    output = U_weights@pattern
    flag = 0
    while np.any(output != -1):
        max_index = np.argmax(output)
        I = np.multiply(D_weights[:,max_index],pattern)
        norm_I = np.count_nonzero(I == 1)
        norm_p = np.count_nonzero(pattern == 1)
        if((norm_I/norm_p)>vigilance):
            #learn
            U_weights[max_index] = (learning_rate + I)/(learning_rate-1+norm_I)
            D_weights[:,max_index] = I
            print(f"winner was cluster {max_index+1}")
            print_pattern(pattern.reshape(8,8))
            flag = 1
            break
        else:
            output[max_index] = -1
    if(flag != 1):
        print("Pattern didn't assigned to a cluster")


