import itertools
import os
import pickle
import time

import networkx as nx
import numpy as np
from joblib import delayed, Parallel


def load(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def edge_in_path(edge, path):
    """
    input:
        - edge: tuple (u, v)
        - path: list of tuple (u, v)
    """
    u, v = edge
    if (u, v) in path or (v, u) in path:
        return True
    return False


class LS2SRSolver:

    def __init__(self, graph, args):
        self.args = args

        # save parameters
        self.G = graph
        self.N = graph.number_of_nodes()
        self.n_edges = len(self.G.edges)
        self.indices_edge = np.arange(self.n_edges)

        self.timeout = args.timeout
        self.verbose = args.verbose

        # compute paths
        self.link2flow = None
        self.compute_path()
        self.ub = self.get_solution_bound(self.flow2link)

        # data for selecting next link -> demand to be mutate
        self.link_selection_prob = None

        # cache
        self.tm = None

    # -----------------------------------------------------------------------------------------------------------------
    def sort_paths(self, paths):
        weights = [[sum(self.G.get_edge_data(u, v)['weight'] for u, v in path)] for path in paths]
        paths = [path for weights, path in sorted(zip(weights, paths), key=lambda x: x[0])]
        return paths

    def get_path(self, i, j, k, paths):
        """
        get a path for flow (i, j) with middle point k
        return:
            - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
        """
        if i == k:
            return None, None

        p_ik = shortest_path(self.G, i, k)
        p_kj = shortest_path(self.G, k, j)
        p = p_ik[:-1] + p_kj

        # remove redundant paths and non-simple path and i == k
        if len(p) != len(set(p)) or p in paths:
            return None, None

        edges = []
        # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
        for u, v in zip(p_ik[:-1], p_ik[1:]):
            edges.append((u, v))
        for u, v in zip(p_kj[:-1], p_kj[1:]):
            edges.append((u, v))

        print('{} {} {}'.format(i, k, j))
        print(edges)
        return edges, p

    def get_paths(self, i, j):
        """
        get all simple path for flow (i, j) on graph G
        return:
            - flows: list of paths
            - path: list of links on path (u, v)
        """
        if i != j:
            path_edges = []
            paths = []
            for k in range(self.N):
                try:
                    edges, path = self.get_path(i, j, k, paths)
                    if edges is not None:
                        path_edges.append(edges)
                        paths.append(path)
                except nx.NetworkXNoPath:
                    pass
            # sort paths by their total link weights for heuristic
            path_edges = self.sort_paths(path_edges)
            return path_edges
        else:
            return []

    def initialize_flow2link(self):
        """
        flow2link is a dictionary:
            - key: flow id (i, j)
            - value: list of paths
            - path: list of links on path (u, v)
        """
        flow2link = {}

        list_paths = Parallel(n_jobs=os.cpu_count())(delayed(self.get_paths)(i, j)
                                                     for i, j in itertools.product(range(self.N), range(self.N)))
        for i, j in itertools.product(range(self.N), range(self.N)):
            flow2link[i, j] = list_paths[i * self.N + j]

        return flow2link

    def initialize_link2flow(self):
        """
        link2flow is a dictionary:
            - key: link id (u, v)
            - value: list of flows id (i, j)
        """
        link2flow = {}
        for u, v in self.G.edges:
            link2flow[(u, v)] = []
        return link2flow

    def compute_path(self):
        folder = os.path.join(self.args.datapath, 'ls2sr/precompute_path')
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, '{}_undirected.pkl'.format(self.args.dataset))
        if os.path.exists(path):
            print('|--- Load precomputed segment from {}'.format(path))
            data = load(path)
            self.link2flow = None
            self.flow2link = data['flow2link']
        else:
            print('|--- Compute segment and save to {}'.format(path))
            self.link2flow = self.initialize_link2flow()
            self.flow2link = self.initialize_flow2link()
            data = {
                'link2flow': self.link2flow,
                'flow2link': self.flow2link,
            }
            save(path, data)

    def get_solution_bound(self, flow2link):
        ub = np.empty([self.N, self.N], dtype=int)
        for i, j in itertools.product(range(self.N), range(self.N)):
            ub[i, j] = len(flow2link[(i, j)])
        ub[ub == 0] = 1
        return ub

    def initialize(self):
        return np.zeros(shape=(self.N, self.N), dtype=int)

    def g(self, i, j, u, v, k):
        if (u, v) in self.flow2link[(i, j)][k] or (v, u) in self.flow2link[(i, j)][k]:
            return 1
        return 0

    def has_path(self, i, j):
        if self.flow2link[(i, j)]:
            return True
        return False

    def set_link_selection_prob(self, alpha=16):
        # compute the prob
        utilizations = nx.get_edge_attributes(self.G, 'utilization').values()
        utilizations = np.array(list(utilizations))
        self.link_selection_prob = utilizations ** alpha / np.sum(utilizations ** alpha)

    def set_flow_selection_prob(self, u, v, beta=1):
        # extract parameters
        tm = self.tm
        # compute the prob
        demands = np.array([tm[i, j] for i, j in self.link2flow[(u, v)]])
        return demands ** beta / np.sum(demands ** beta)

    def select_flow(self):
        # select link
        self.set_link_selection_prob()
        idx_sort = np.argsort(self.link_selection_prob)[-int(0.2 * len(self.indices_edge)):]
        indices_edge = self.indices_edge[idx_sort]
        index = np.random.choice(indices_edge)
        link = list(self.G.edges)[index]

        # select flow
        flow_prob = self.set_flow_selection_prob(link[0], link[1])
        indices = np.arange(len(self.link2flow[link]))

        index = np.random.choice(indices, p=flow_prob)
        flow = self.link2flow[link][index]
        return flow

    def set_link2flow(self, solution):
        # extract parameters
        # initialize link2flow
        self.link2flow = {}
        for edge in self.G.edges:
            self.link2flow[edge] = []
        # compute link2flow
        for edge in self.G.edges:
            for i, j in self.flow2link:
                k = solution[i, j]
                if self.has_path(i, j):
                    path = self.flow2link[i, j][k]
                    if edge_in_path(edge, path):
                        self.link2flow[edge].append((i, j))

    def evaluate(self, solution, tm=None):
        # extract parameters
        if tm is None:
            tm = self.tm
        # extract utilization
        utilizations = []
        for u, v in self.G.edges:
            load = 0
            for i, j in itertools.product(range(self.N), range(self.N)):
                if self.has_path(i, j):
                    k = solution[i, j]
                    load += self.g(i, j, u, v, k) * tm[i, j]
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization
            utilizations.append(utilization)
        return max(utilizations)

    def evaluate_fast(self, new_path_idx, best_solution, i, j):
        # get current utilization from edges
        utilizations = nx.get_edge_attributes(self.G, 'utilization')

        # new solution
        new_path = self.flow2link[(i, j)][new_path_idx]

        # current best solution
        best_path_idx = best_solution[i, j]
        best_path = self.flow2link[(i, j)][best_path_idx]

        # accumulate the utilization
        for u, v in best_path:
            u, v = sorted((u, v))
            utilizations[(u, v)] -= self.tm[i, j] / self.G[u][v]['capacity']
        for u, v in new_path:
            u, v = sorted((u, v))
            utilizations[(u, v)] += self.tm[i, j] / self.G[u][v]['capacity']

        return utilizations

    def update_link2flows(self, old_path_idx, new_path_idx, i, j):
        """
        Updating link2flows after changing path of flow (i,j)
        """
        old_path = self.flow2link[i, j][old_path_idx]
        new_path = self.flow2link[i, j][new_path_idx]

        for edge in self.G.edges:
            if edge_in_path(edge, old_path):
                self.link2flow[edge].remove((i, j))
            if edge_in_path(edge, new_path):
                self.link2flow[edge].append((i, j))

    def apply_solution(self, utilizations):
        nx.set_edge_attributes(self.G, utilizations, name='utilization')

    def solve(self, tm, solution=None, eps=1e-8):
        # save parameters
        self.tm = tm

        # initialize solution
        if solution is None:
            solution = self.initialize()

        # initialize solver state
        self.set_link2flow(solution)
        best_solution = solution
        u = self.evaluate(solution)
        theta = u

        # iteratively solve
        tic = time.time()
        while time.time() - tic < self.timeout:
            i, j = self.select_flow()
            if i == j:
                continue
            new_path_idx = best_solution[i, j] + 1
            if new_path_idx >= self.ub[i, j]:
                new_path_idx = 0

            utilization = self.evaluate_fast(new_path_idx, best_solution, i, j)
            mlu = max(utilization.values())
            if theta - mlu >= eps:
                self.update_link2flows(old_path_idx=best_solution[i, j], new_path_idx=new_path_idx, i=i, j=j)
                self.apply_solution(utilization)  # updating utilization in Graph aka self.G
                best_solution[i, j] = new_path_idx
                theta = mlu
        return best_solution