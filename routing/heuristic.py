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
    sorted_edge = tuple(sorted(edge))
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
                    if path:  # if there exists path
                        # in other word, path is not []
                        if is_simple_path(path):
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
        G = self.G
        # select link
        indices = np.arange(len(G.edges))
        index = np.random.choice(indices, p=self.link_selection_prob)
        link = list(G.edges)[index]
        # select flow
        indices = np.arange(len(self.link2flow[link]))
        index = np.random.choice(indices, p=self.demand_selection_prob[link])
        flow = self.link2flow[link][index]
        return flow

    def set_link2flow(self, solution):
        # extract parameters
        G = self.G
        # initialize link2flow
        self.link2flow = {}
        for edge in G.edges:
            self.link2flow[edge] = []
        # compute link2flow
        for edge in G.edges:
            for i, j in self.flow2link:
                k = solution[i, j]
                if self.has_path(i, j):
                    path = self.flow2link[i, j][k]
                    if edge_in_path(edge, path):
                        self.link2flow[(edge)].append((i, j))

    def set_lowerbound(self, solution):
        self.lb = solution.copy()

    def set_G(self, G):
        self.G = G

    def evaluate(self, solution, tm=None, save_utilization=False):
        # extract parameters
        if save_utilization:
            G = self.G
        else:
            G = self.G.copy()
        N = G.number_of_nodes()
        if tm is None:
            tm = self.tm
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = 0
            demands = []
            for i, j in itertools.product(range(N), range(N)):
                if self.has_path(i, j):
                    k = solution[i, j]
                    load += self.g(i, j, u, v, k) * tm[i, j]
                    if self.g(i, j, u, v, k):
                        demands.append((i, j))
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization
            utilizations.append(utilization)
        return max(utilizations)

    def mutate(self, solution, i, j):
        self.lb[i, j] = self.lb[i, j] + 1
        if self.lb[i, j] >= self.ub[i, j]:
            self.lb[i, j] = 0
        solution[i, j] = self.lb[i, j]
        return solution

    def evaluate_fast(self, solution, best_solution, i, j):
        # extract parameters
        G = self.G.copy()
        tm = self.tm

        # extract old and new path
        k = solution[i, j]
        best_k = best_solution[i, j]
        path = self.flow2link[(i, j)][k]
        best_path = self.flow2link[(i, j)][best_k]

        # accumulate the utilization
        for u, v in best_path:
            u, v = sorted([u, v])
            G[u][v]['utilization'] -= tm[i, j] / G[u][v]['capacity']
        for u, v in path:
            u, v = sorted([u, v])
            G[u][v]['utilization'] += tm[i, j] / G[u][v]['capacity']
        # get all utilizations from edges
        utilizations = nx.get_edge_attributes(G, 'utilization').values()
        return max(utilizations), G

    def solve(self, tm, solution=None, eps=1e-6):
        # save parameters
        self.tm = tm

        # initialize solution
        if solution is None:
            solution = self.initialize()

        # initialize solver state
        self.set_link2flow(solution)
        best_solution = solution.copy()
        u = self.evaluate(solution, save_utilization=True)
        theta = u
        self.set_link2flow(best_solution)
        self.set_link_selection_prob()
        self.set_flow_selection_prob()
        self.set_lowerbound(best_solution)
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
            u, G = self.evaluate_fast(solution, best_solution, i, j)
            # u_exact = self.evaluate(solution)
            # np.testing.assert_almost_equal(u, u_exact, decimal=6)
            if u - theta < -eps:
                best_solution = solution.copy()
                u_exact = self.evaluate(best_solution, save_utilization=True)
                np.testing.assert_almost_equal(u, u_exact, decimal=6)
                theta = u_exact
                self.set_link2flow(best_solution)
                self.set_link_selection_prob()
                self.set_flow_selection_prob()
                self.set_lowerbound(best_solution)
                self.set_G(G)
                if self.verbose:
                    print('[+] new solution found n={} t={:0.2f} i={} j={} tm={:0.2f} theta={}'.format(
                        num_eval, time.time() - tic, i, j, tm[i, j], theta))
        if self.verbose:
            print('[+] final solution: n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                num_eval, time.time() - tic, i, j, tm[i, j], theta))
        return best_solution
