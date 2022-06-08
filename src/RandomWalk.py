"""Original Random Walk method. """

import random
import multiprocessing as mp


class RandomWalker:
    """Random Walk method

    Args:
        graph (nx.Graph): The graph is built from networkx package.
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
        walks = []

        for start_node in nodelist:
            walk = self.random_walk(walk_len, start_node)
            walks.append(walk)

        return walks

    def random_walk(self, walk_len, start_node):
        walk = [start_node]

        while len(walk) < walk_len:
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))

            if len(cur_neighbors) > 0:
                walk.append(random.choice(cur_neighbors))
            else:  # may happen at directed graph
                break
        return walk
