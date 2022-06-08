"""Efficient Random Walk method.
Adjust number of walk according to node's appearance times
"""

import random
from collections import Counter
import numpy as np
import multiprocessing as mp


class RandomWalker:
    """Random Walk method

    Args:
        graph (nx.Graph): The graph is built from networkx package.
        num_walks (int): initial number of walks.
        graph_type (str): 'sparse' or 'dense' or 'None'.
    """

    def __init__(self, graph, num_walks):
        self.graph = graph
        self.num_walks = num_walks
        self.nodelist = list(graph.nodes()) * num_walks
        random.shuffle(self.nodelist)  ##

    def multi_walk(self, walk_len, n_workers):
        pool = mp.Pool(n_workers)

        split_list = lambda lst, n: [lst[i::n] for i in range(n)]
        node_lists = split_list(self.nodelist, n_workers)

        results, tmp = [], []
        for node_list in node_lists:
            result = pool.apply_async(self._multi_walk, args=(node_list, walk_len))
            tmp.append(result)
        pool.close()
        pool.join()

        for r in tmp:
            results += r.get()

        print("num_walks: ", len(results))  ##
        return results

    def _multi_walk(self, nodelist, walk_len):
        node_counter = Counter(nodelist)
        walks = []

        while len(node_counter) > 0:
            start_node = random.choice(list(node_counter.keys()))
            walk = self.random_walk(walk_len, start_node)
            walks.append(walk)

            cnt_node = Counter(walk)
            node_counter.subtract(cnt_node)
            node_counter += Counter()  # Remove cnt equal to zero or negative

        return walks

    def random_walk(self, walk_len, start_node):
        walk = [start_node]

        while len(walk) < walk_len:
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))

            if len(cur_neighbors) > 0:
                walk.append(random.choice(cur_neighbors))
            else:
                break
        return walk
