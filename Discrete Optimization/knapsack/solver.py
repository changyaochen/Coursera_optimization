#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
from collections import namedtuple
from typing import List
Item = namedtuple("Item", ['index', 'value', 'weight'])


class BranchAndBound(object):
    """Branch and bound algo for 0/1 knapsack.

    We will use dfs with early stopping
    """

    def __init__(
        self, 
        item_count: int,
        capacity: int,
        lines: List[str],
        timeout: int = 600):

        sys.setrecursionlimit(10000)  # default is 1,000
       
        self.item_count = item_count
        self.capacity = capacity
        self.items = self._parse_line(item_count, lines)
        self.timeout = timeout
        self.t_start = None
        
        self.values = [x.value for x in self.items]
        self.weights = [x.weight for x in self.items]
        self.current_max = 0
        self.possible_max = [sum(self.values)] * self.item_count
        self.taken = [0] * self.item_count

        # let's take the greedy method
        self.back_mapping = {i: i for i in range(self.item_count)}
        self._sort_by_density()
        # self._sort_by_weight()
        print(self.values)
        print(self.possible_max)

    def _parse_line(self, item_count: int, lines: List[str]):

        items = []
        for i in range(1, item_count + 1):
            line = lines[i]
            parts = line.split()
            items.append(Item(i - 1, int(parts[0]), int(parts[1])))

        return items

    def _sort_by_density(self):
        """Sort the items by value/weigth ratio"""
        lst = [(i, v, w, 1. * v / w) for i, v, w in zip(
            list(range(self.item_count)), self.values, self.weights)]
        lst = sorted(lst, key=lambda x: x[3], reverse=True)
        self.values = [x[1] for x in lst]
        self.weights = [x[2] for x in lst]
        for i in range(self.item_count):
            self.back_mapping[i] = lst[i][0]
        
        def _get_possible_max(values, weights, capacity):
            # calculate maximum possible value
            total_value, total_weight = 0, 0
            for i in range(len(values) - 1):
                if total_weight < capacity:
                    total_weight += weights[i]
                    total_value += values[i]
            # add one last item, as a whole
            return total_value + values[i]

        for i in range(self.item_count - 1):
            self.possible_max[i] = _get_possible_max(
                self.values[i:],
                self.weights[i:],
                self.capacity
            )

    def _sort_by_weight(self):
        """Sort the items by weight"""
        lst = [(i, v, w) for i, v, w in zip(
            list(range(self.item_count)), self.values, self.weights)]
        lst = sorted(lst, key=lambda x: x[2])
        self.values = [x[1] for x in lst]
        self.weights = [x[2] for x in lst]
        for i in range(self.item_count):
            self.back_mapping[i] = lst[i][0]

    def dfs(self, 
            idx: int, remaining_cap: int, remaining_value: int,
            values: List, weights: List,
            current_taken: List, current_value: int):
        
        # timeout setting
        if time.time() - self.t_start > self.timeout:
            # print('Stopped due to time limit of {} seconds.'
            #     .format(self.timeout))
            return

        if len(values) == 0:
            # base case, no more item
            # print('No item left.')
            return
        
        # if remaining_value + current_value < self.current_max:
        if self.possible_max[idx] + current_value < self.current_max:  
            # early stopping
            # print('The best possible solution is worse than existing one.')
            return

        # recursive calls
        # let's consider this item
        value, weight = values[0], weights[0] 
        remaining_value -= value
        
        # option 1, take it, if allowed
        if (remaining_cap - weight) >= 0:  # there is capacity left for it
            # print('\n===> In option 1, idx ', idx)
            current_taken[idx] = 1
            if current_value + value >= self.current_max:  # better solution
                self.current_max = current_value + value
                self.taken = current_taken[:]
                # print('\nfound a better solution of {}'
                #       .format(current_value + value))
                # print('taken items {}'.format(self.taken))
            # keep going
            self.dfs(idx + 1, remaining_cap - weight, remaining_value,
                     values[1:], weights[1:], 
                     current_taken, current_value + value)

        # option 2, don't take it
        # print('\n===> In opition 2, idx ', idx)
        # print('remaining value: ', remaining_value)
        # print('remaining cap: ', remaining_cap)
        current_taken[idx] = 0
        self.dfs(idx + 1, remaining_cap, remaining_value,
                 values[1:], weights[1:], 
                 current_taken, current_value)

    def run(self):

        self.t_start = time.time()
        init_capacity = self.capacity
        remaining_value = sum(self.values)
        init_values = self.values
        init_weights = self.weights
        init_taken = [0] * self.item_count

        self.dfs(0, init_capacity, remaining_value,
                 init_values, init_weights, init_taken, 0)

        # remap back to the original index
        new_taken = [0] * self.item_count
        for i in range(self.item_count):
            new_taken[self.back_mapping[i]] = self.taken[i]
        self.taken = new_taken


def _parse_line(item_count: int, lines: List[str]):

    items = []
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return items


def naive_solver(
    item_count: int,
    capacity: int,
    lines: List[str]):
    
    items = _parse_line(item_count, lines)

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


def dp_solver(
    item_count: int,
    capacity: int,
    lines: List[str]):
    """Dynamic Programming solution."""
    
    items = _parse_line(item_count, lines)
    taken = [0] * len(items)

    # matrix for capacity (r) x item_count (c)
    n_cols = (1 + item_count)
    n_rows = capacity + 1
    # print('initing dp matrix...')
    dp = np.zeros((n_rows, n_cols), np.int32)
    # print('inited!')

    # let me fill column by column
    for i in range(1, n_cols):
        # going row by row
        # print('{} of {}'.format(i, n_cols))
        for j in range(1, n_rows):
            if items[i - 1].weight <= j:  # the capacity is at least big enough
                dp[j][i] = max(
                    # take i-th item
                    items[i - 1].value + \
                    dp[max(0, j - items[i - 1].weight)][i - 1],  
                    # do not take i-th item
                    dp[j][i - 1]   
                )
            else:  # can take this item
                dp[j][i] = dp[j][i - 1]
    
    value = dp[-1][-1]
    # print(dp)
    # print(value)

    # retrace the decision
    j = capacity
    for i in range(n_cols - 1, 0, -1):
        if dp[j][i] == dp[j][i - 1]:  # not taken i-th item
            pass
        else:  # we took i-th item
            taken[i - 1] = 1
            j -= items[i - 1].weight

    return value, taken


def dp_solver_2(
    item_count: int,
    capacity: int,
    lines: List[str]):
    """Dynamic Programming solution, with less space"""
    
    items = _parse_line(item_count, lines)
    taken = [0] * item_count

    # matrix for capacity (r) x item_count (c)
    n_cols = (1 + item_count)
    n_rows = capacity + 1
    print('initing dp array')
    dp_old = [0] * n_rows
    dp_new = [0] * n_rows
    print('inited!')

    for i in range(item_count):
        # print('{} of {}'.format(i, item_count))
        for j in range(1, n_rows):
            if items[i].weight > j:  # can take this item
                dp_new[j] = dp_old[j]
            else:  # we are allow to take this item
                possible_value = items[i].value + \
                    dp_old[max(0, j - items[i].weight)]
                if possible_value >= dp_old[j]: 
                    # we will take item i
                    dp_new[j] = possible_value
                else:
                    # do not take this item
                    dp_new[j] = dp_old[j]
        # done with one pass, update
        dp_old = dp_new[:]
        dp_new = [0] * n_rows

    value = dp_old[-1]

    return value, taken


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    if item_count * capacity <= 0e10:
        print('Using DP...')
        value, taken = dp_solver(item_count, capacity, lines)
    else:
        print('Using branch and bound')
        bb = BranchAndBound(item_count, capacity, lines)
        bb.run()
        value, taken = bb.current_max, bb.taken
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import os

    start = time.time()
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        file_locations = [file_location]
    else:  # test cases
        # print('This test requires an input file.  Please select one from the'
        #       ' data directory. (i.e. python solver.py ./data/ks_4_0)')
        file_locations = ['./data/' + x for x in os.listdir('./data')]
        file_locations = [
            './data/ks_4_0', 
            # './data/ks_lecture_dp_2', 
            # './data/ks_19_0', 
            # './data/ks_30_0',
            # './data/ks_50_0'
        ]
        answers = [
            19, 
            # 44,
            # 12248, 
            # 99798,
            # 142156
        ]

    for i, f in enumerate(file_locations):
        print(f)
        with open(f, 'r') as input_data_file:
            input_data = input_data_file.read()
        tmp = solve_it(input_data).split('\n')
        value = int(tmp[0].split(' ')[0])
        decision = tmp[1]
        print(value)
        print(decision)

        if not answers[i] == value:
            print("!!! ===== FAILED ===== !!!\n"
                  "The best value is {}, but seen {}"
                  .format(answers[i], value))
        print('Time taken (seconds): {0:5.3f}'.format(time.time() - start))
