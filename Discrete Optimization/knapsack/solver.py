#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import List
Item = namedtuple("Item", ['index', 'value', 'weight'])


def _parse_line(item_count: int, lines: List[str]):

    items = []
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return items


def naive_solver(item_count: int,
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


def dp_solver(item_count: int,
              capacity: int,
              lines: List[str]):
    """Dynamic Programming solution."""
    
    items = _parse_line(item_count, lines)
    taken = [0] * len(items)

    # matrix for capacity (r) x item_count (c)
    n_cols = (1 + item_count)
    n_rows = capacity + 1
    dp = [[0] * n_cols for _ in range(n_rows)]

    # let me fill column by column
    for i in range(1, n_cols):
        # going row by row
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


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    value, taken = dp_solver(item_count, capacity, lines)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        file_locations = [file_location]
    else:
        # print('This test requires an input file.  Please select one from the'
        #       ' data directory. (i.e. python solver.py ./data/ks_4_0)')
        file_locations = ['./data/' + x for x in os.listdir('./data')]

    for f in file_locations:
        print(f)
        with open(f, 'r') as input_data_file:
                input_data = input_data_file.read()
        print(solve_it(input_data))
