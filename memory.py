#### Code modified from https://github.com/abderraouf2che/RL-Traffic-Signal-Control/blob/main/Memory.py  #####

import random
import numpy as np

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01 # A small positive value to avoid division by zero and losing some experience
    a = 0.6
    beta = 0.4 # The higher the beta, the more priority is given to samples with high errors.
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, size_min):
        self.tree = SumTree(capacity) # An object for storing experiences with priorities
        self.capacity = capacity # memory capacity
        self._size_min = size_min # min amount of accumulated experience before sampling
        
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a # higher priorities to experiences with larger error

    def add_sample(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def get_samples(self, n):
        # if self._size_now() < self._size_min:
        #     return []

        # if n > self._size_now():
        #     return random.sample(self._samples, self._size_now())  # get all the samples
        # else:   
        batch = [] # Stores the retrieved data samples.
        idxs = [] # Stores the indexes of data samples in the SumTree.
        segment = self.tree.total() / n # E.g: Size of each segment = 10 (total priorities) / 5 (segment) = 2 (priorities each segment)
        priorities = []
    
        # plus beta_increment_per_sampling to avoid the model only learns the highest priority experience
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # limit beta <= 1

        for i in range(n):
            # start (a) and end (b) points for the current segment
            a = segment * i
            b = segment * (i + 1)
            # E.g: 2*0 = 0 -> 2*1 = 2

            s = random.uniform(a, b) # random value is within the a-b range
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
    
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta) # importance sampling weight
        is_weight /= is_weight.max() # normalize importance sampling weight in range 0-1
    
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    # def _size_now(self):
    #     """
    #     Check how full the memory is
    #     """
    #     print("tree:", np.array(self.tree).shape))
    #     return int(np.array(self.tree).shape)

# Priorities is stored in a binary tree implemented on arrays, data is stored in a normal array
# Leaf nodes store the experiences and their corresponding priorities.
# Non-leaf nodes (parent nodes) store the sum of the priorities of all their children (subordinate nodes).
class SumTree:
    write = 0 # The pointer indicates the location to save the next sample into data array.

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # a binary tree implemented on the array, used to stores priority values
        self.data = np.zeros(capacity, dtype=object) # normal array used to store data
        self.n_entries = 0 # number of data entries in the tree

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2 # index of the parent in the array = (index of the child - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1 # left child of current node
        right = left + 1 # right child of current node

        # Check if the left child is outside the self.tree storage array.
        # If yes, the current node (idx) is already a leaf node and the function returns this index.
        if left >= len(self.tree): 
            return idx

        # If the random value s <= priority value of the left child, traversal to the left child because the data may be there.
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        
        # E.g: Parent = 10, child = 4, 6
        # s = 5 > 4 => s = 5 - 4 = 1 and call the function again with right child node
        else:
            return self._retrieve(right, s - self.tree[left])

    # Because in SumTree, the parent node store sum of all child nodes, the total is stored in the first parent node.
    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        
        # E.g.: capacity (maximum amount of experience) = 10
        # write = 0 -> 9 => idx of priority child nodes = 9 -> 18 
        # => index of priority parent nodes (sum of child nodes) = 0 -> 8 in self.tree (binary tree implemented on array)
        # index of data = 0 -> 9 in self.data (normal array)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s) 
        dataIdx = idx - self.capacity + 1 
        # Convert index of priority in self.tree array to corresponding index of data in self.data array

        return (idx, self.tree[idx], self.data[dataIdx])


# import heapq
# import torch
# from itertools import count
# from collections import deque
# tiebreaker = count()

# class Memory:
#     def __init__(self, size_max, size_min):
#         self._samples = []
#         self._size_max = size_max
#         self._size_min = size_min


#     def add_sample(self, TD, transition):
#         """
#         Add a sample into the memory
#         """
#         # self._samples.append(transition)
#         heapq.heappush(self._samples, (-TD, next(tiebreaker), transition))
#         heapq.heapify(self._samples)
#         if self._size_now() > self._size_max:
#             self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element


#     def get_samples(self, n, model,train=0):
#         """
#         Get n samples randomly from the memory
#         """
#         if self._size_now() < self._size_min:
#             return []

#         if n > self._size_now():
#             return random.sample(self._samples, self._size_now())  # get all the samples
#         else:
#             x = random.sample(self._samples, 10*n)  # get "batch size" number of samples
#             # if self._size_now() > self._size_max:
#             #     self._samples = self._samples[:-1]
#             batch = heapq.nsmallest(n, x)
#             batch = [e for (_, _, e) in batch]
#             # print(batch)

#             del self._samples[0:n]
#             # self._samples = self._samples[n:]
            
#             return batch


#     def _size_now(self):
#         """
#         Check how full the memory is
#         """
#         return len(self._samples)