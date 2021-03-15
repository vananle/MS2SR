import itertools
import time

import networkx as nx
import numpy as np


def shortest_path(G, source, target):
    return nx.shortest_path(G, source=source, target=target, weight='weight')


def remove_duplicated_path(paths):
    # convert path to string
    paths = ['-'.join('{}/{}'.format(u, v) for u, v in path) for path in paths]
    # remove duplicated string
    paths = list(set(paths))
    # convert string to path
    new_paths = []
    for path in paths:
        new_path = []
        for edge_str in path.split('-'):
            u, v = edge_str.split('/')
            u, v = int(u), int(v)
            new_path.append((u, v))
        new_paths.append(new_path)
    return new_paths


def is_simple_path(path):
    '''
    input:
        - path: which is a list of edges (u, v)
    return:
        - is_simple_path: bool
    '''
    edges = []
    for edge in path:
        edge = tuple(sorted(edge))
        if edge in edges:
            return False
        edges.append(edge)
    return True


def get_path(G, i, j, k):
    '''
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path
    '''
    p_ik = shortest_path(G, i, k)
    p_kj = shortest_path(G, k, j)
    edges = []
    # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
    if len(p_ik) > 1:
        for u, v in zip(p_ik[:-1], p_ik[1:]):
            edges.append((u, v))
        for u, v in zip(p_kj[:-1], p_kj[1:]):
            edges.append((u, v))
    return edges


def edge_in_path(edge, path):
    '''
    input:
        - edge: tuple (u, v)
        - path: list of tuple (u, v)
    '''
    sorted_path_edges = [tuple(sorted(path_edge)) for path_edge in path]
    if edge in sorted_path_edges:
        return True
    return False


class HeuristicSolver:

    def __init__(self, G, time_limit=10, verbose=False):
        # save parameters
        self.G = G
        self.N = G.number_of_nodes()
        self.time_limit = time_limit
        self.verbose = verbose

        # compute paths
        self.link2flow = None
        self.flow2link = self.initialize_flow2link()
        self.lb, self.ub = self.get_solution_bound(self.flow2link)

        # data for selecting next link -> demand to be mutate
        self.link_selection_prob = None
        self.demand_selection_prob = None

        # cache
        self.tm = None

    # -----------------------------------------------------------------------------------------------------------------
    def sort_paths(self, paths):
        weights = [[sum(self.G.get_edge_data(u, v)['weight'] for u, v in path)] for path in paths]
        paths = [path for weights, path in sorted(zip(weights, paths), key=lambda x: x[0])]
        return paths

    def get_paths(self, i, j):
        '''
        get all simple path for flow (i, j) on graph G
        return:
            - flows: list of paths
            - path: list of links on path (u, v)
        '''
        paths = []
        if i != j:
            for k in range(self.N):
                try:
                    path = get_path(self.G, i, j, k)
                    if path and is_simple_path(path):
                        paths.append(path)
                except nx.NetworkXNoPath:
                    pass
            # remove redundant paths
            paths = remove_duplicated_path(paths)
            # sort paths by their total link weights for heuristic
            paths = self.sort_paths(paths)
        return paths

    def initialize_link2flow(self):
        '''
        link2flow is a dictionary:
            - key: link id (u, v)
            - value: list of flows id (i, j)
        '''
        link2flow = {}
        for u, v in self.G.edges:
            link2flow[(u, v)] = []
        return link2flow

    def initialize_flow2link(self):
        '''
        flow2link is a dictionary:
            - key: flow id (i, j)
            - value: list of paths
            - path: list of links on path (u, v)
        '''
        flow2link = {}
        for i, j in itertools.product(range(self.N), range(self.N)):
            paths = self.get_paths(i, j)
            flow2link[i, j] = paths
        return flow2link

    def get_solution_bound(self, flow2link):
        lb = np.zeros([self.N, self.N], dtype=int)
        ub = np.empty([self.N, self.N], dtype=int)
        for i, j in itertools.product(range(self.N), range(self.N)):
            ub[i, j] = len(flow2link[(i, j)])
        ub[ub == 0] = 1
        return lb, ub

    def initialize(self):
        return np.zeros_like(self.lb)

    def g(self, i, j, u, v, k):
        if (u, v) in self.flow2link[(i, j)][k] or \
                (v, u) in self.flow2link[(i, j)][k]:
            return 1
        return 0

    def has_path(self, i, j):
        if self.flow2link[(i, j)]:
            return True
        return False

    def set_link_selection_prob(self, alpha=16):
        # extract parameters
        G = self.G
        # compute the prob
        utilizations = nx.get_edge_attributes(G, 'utilization').values()
        utilizations = np.array(list(utilizations))
        self.link_selection_prob = utilizations ** alpha / np.sum(utilizations ** alpha)

    def set_flow_selection_prob(self, alpha=1):
        # extract parameters
        G = self.G
        tm = self.tm
        # compute the prob
        self.demand_selection_prob = {}
        for u, v in G.edges:
            demands = np.array([tm[i, j] for i, j in self.link2flow[(u, v)]])
            self.demand_selection_prob[(u, v)] = demands ** alpha / np.sum(demands ** alpha)

    def select_flow(self):
        # extract parameters
        # select link
        indices = np.arange(len(self.G.edges))
        index = np.random.choice(indices, p=self.link_selection_prob)
        link = list(self.G.edges)[index]
        # select flow
        indices = np.arange(len(self.link2flow[link]))
        index = np.random.choice(indices, p=self.demand_selection_prob[link])
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
            # demands = [] # ???
            for i, j in itertools.product(range(self.N), range(self.N)):
                if self.has_path(i, j):
                    k = solution[i, j]
                    load += self.g(i, j, u, v, k) * tm[i, j]
                    # todo: explain these two lines
                    # if self.g(i, j, u, v, k):  # ???
                    #     demands.append((i, j))
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization
            utilizations.append(utilization)
        return max(utilizations)

    def mutate(self, solution, i, j):
        self.lb[i, j] = self.lb[i, j] + 1
        if self.lb[i, j] >= self.ub[i, j]:
            self.lb[i, j] = 0
        solution[i, j] = self.lb[i, j]
        return solution

    def evaluate_fast(self, solution, best_solution, i, j):
        # get current utilization from edges
        _utilizations = nx.get_edge_attributes(self.G, 'utilization')

        # new solution
        path_idx = solution[i, j]
        path = self.flow2link[(i, j)][path_idx]

        # current best solution
        best_path_idx = best_solution[i, j]
        best_path = self.flow2link[(i, j)][best_path_idx]

        # accumulate the utilization
        for u, v in best_path:
            u, v = sorted([u, v])
            _utilizations[(u, v)] -= self.tm[i, j] / self.G[u][v]['capacity']
        for u, v in path:
            u, v = sorted([u, v])
            _utilizations[(u, v)] += self.tm[i, j] / self.G[u][v]['capacity']
        # utilizations = nx.get_edge_attributes(G, 'utilization').values()

        return max(_utilizations)

    def update_link2flows(self, solution, new_solution, i, j):
        """
        Updating link2flows after changing path of flow (i,j)
        """
        old_path_idx = solution[i, j]
        old_path = self.flow2link[i, j][old_path_idx]
        new_path_idx = new_solution[i, j]
        new_path = self.flow2link[i, j][new_path_idx]

        for edge in self.G.edges:
            if edge_in_path(edge, old_path):
                self.link2flow[edge].remove((i, j))
            if edge_in_path(edge, new_path):
                self.link2flow[edge].append((i, j))

    def apply_solution(self, utilizations):
        nx.set_edge_attributes(self.G, utilizations, name='utilization')

        # todo: remove this line
        assert np.array_equal(nx.get_edge_attributes(self.G, 'utilization').values(), utilizations)

    def solve(self, tm, solution=None, eps=1e-6):
        # save parameters
        self.tm = tm

        # initialize solution
        if solution is None:
            solution = self.initialize()

        # initialize solver state
        self.set_link2flow(solution)
        best_solution = solution.copy()
        u = self.evaluate(solution)
        theta = u
        self.set_link2flow(best_solution)
        self.set_link_selection_prob()
        self.set_flow_selection_prob()
        self.lb = np.copy(best_solution)
        tic = time.time()

        if self.verbose:
            print('initial theta={}'.format(u))

        # iteratively solve
        num_eval = 0
        while time.time() - tic < self.time_limit:
            num_eval += 1
            i, j = self.select_flow()
            solution = best_solution.copy()
            solution = self.mutate(solution, i, j)
            utilization = self.evaluate_fast(solution, best_solution, i, j)
            mlu = np.max(utilization)
            if mlu - theta < -eps:
                # -------- Applying the better solution ---------------
                # updating link2flow for flow (i,j)
                self.update_link2flows(solution=best_solution, new_solution=solution, i=i, j=j)
                self.apply_solution(utilization)  # updating utilization in Graph aka self.G
                self.set_link_selection_prob()
                self.set_flow_selection_prob()
                best_solution = solution
                theta = mlu

                # self.set_lowerbound(best_solution) -->  self.lb[i,j] = best_solution[i,j]
                self.lb[i, j] = best_solution[i, j]
                if self.verbose:
                    print('[+] new solution found n={} t={:0.2f} i={} j={} tm={:0.2f} theta={}'.format(
                        num_eval, time.time() - tic, i, j, tm[i, j], theta))
        if self.verbose:
            print('[+] final solution: n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                num_eval, time.time() - tic, i, j, tm[i, j], theta))
        return best_solution
