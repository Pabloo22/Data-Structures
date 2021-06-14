import numpy as np
from copy import deepcopy
from typing import TypeVar
from priority_queue import PriorityQueue
from collections import deque
from random import choice
from itertools import combinations

Matrix = TypeVar('Matrix', list[list[float]], np.ndarray)
AdjList = list[list[tuple[float, int]]]


class Graph:

    __size: int
    __adj_list: list[list[tuple[float, int]]]
    __adj_matrix: None or Matrix
    __labels: list

    def __init__(self, n_nodes: int = 0,
                 labels: list = None,
                 adj_list: AdjList = None,
                 adj_matrix=None,
                 build_matrix=False):

        if isinstance(labels, list):
            if len(labels) != n_nodes:
                raise ValueError("len(labels) != n_nodes")

        self.__size = n_nodes
        self.__adj_list = [[] for _ in range(n_nodes)] if adj_list is None else adj_list
        self.__labels = [None] * n_nodes if labels is None else labels

        if adj_matrix is None:
            if not build_matrix:
                self.__adj_matrix = None
            else:
                self.__adj_matrix = Graph.create_adj_matrix(self.__adj_list)
        else:
            self.__adj_matrix = adj_matrix
            self.__adj_list = Graph.create_adj_list(adj_matrix) if adj_list is None else adj_list

    @property
    def adj_list(self):
        return self.__adj_list

    @property
    def size(self):
        return self.__size

    @staticmethod
    def create_adj_matrix(adj_list: AdjList) -> Matrix:
        adj_matrix = np.zeros((len(adj_list), len(adj_list)), dtype=float)
        for i, value in enumerate(adj_list):
            for w, j in value:
                adj_matrix[i][j] = w

        return adj_matrix

    @staticmethod
    def create_adj_list(adj_matrix: Matrix) -> AdjList:
        adj_list = [[] for _ in range(len(adj_matrix))]
        for i, row in enumerate(adj_matrix):
            for j, w in enumerate(row):
                if w:
                    adj_list[i].append((w, j))

        return adj_list

    @property
    def adj_matrix(self):
        if self.__adj_matrix is None:
            self.__adj_matrix = Graph.create_adj_matrix(self.__adj_list)

        return self.__adj_matrix
        
    @property
    def labels(self):
        return self.__labels

    def edges(self) -> list[tuple[float, tuple[int, int]]]:
        edges = []
        for i, L in enumerate(self.__adj_list):
            for w, j in L:
                if (w, (j, i)) not in edges:
                    edges.append((w, (i, j)))

        return edges

    def weight(self):
        return sum(map(lambda x: x[0], self.edges()))

    def add_edge(self, i: int, j: int, weight: float = 1):
        if self.__adj_matrix is not None:
            self.adj_matrix[i][j] = weight
            self.adj_matrix[j][i] = weight

        self.__adj_list[i].append((weight, j))
        self.__adj_list[j].append((weight, i))

    def add_arc(self, i: int, j: int, weight: float = 1):
        if self.__adj_matrix is not None:
            self.adj_matrix[i][j] = weight

        self.__adj_list[i].append((weight, j))

    def set_label(self, i: int, value: any):
        if self.__adj_matrix is not None:
            self.__labels[i] = value

    def get_label(self, i: int) -> any:
        return self.__labels[i]

    def adjacent_to(self, i: int) -> list[tuple[float, int]]:
        return self.__adj_list[i]

    def is_a_path_from_to(self, i, j):
        transitive_closure = self.warshall()
        return transitive_closure[i][j]

    def is_connected(self):
        transitive_closure = self.warshall()
        for row in transitive_closure:
            if 0 in row:
                return False
        return True

    def warshall(self) -> Matrix:
        """ O(V^3)"""
        n = len(self.adj_matrix)
        transitive_matrix = deepcopy(self.adj_matrix)
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    if transitive_matrix[i][j] or (transitive_matrix[i][k] and transitive_matrix[k][j]):
                        transitive_matrix[i][j] = 1
                    else:
                        transitive_matrix[i][j] = 0

        return transitive_matrix

    def floyd(self) -> Matrix:
        """O(V^3)"""

        n = len(self.adj_matrix)
        dist_matrix = deepcopy(self.adj_matrix)
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    if dist_matrix[i][j] or (dist_matrix[i][k] and dist_matrix[k][j]) and i != j:
                        if not dist_matrix[i][j]:
                            dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                        elif dist_matrix[i][k] and dist_matrix[k][j]:
                            dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
        return dist_matrix

    def kruskal(self) -> 'Graph':
        """O(E log V)"""
        adj_list: AdjList = [[] for _ in range(self.__size)]
        n_components = self.__size
        parent_array: list[int] = list(range(self.__size))

        edges = self.edges()
        pq = PriorityQueue(edges)

        while n_components > 1 and pq.size() > 0:
            w, e = pq.dequeue()
            i, j = e
            parent_i = Graph._find_set(i, parent_array)
            parent_j = Graph._find_set(j, parent_array)
            if parent_j != parent_i:
                Graph._join_set(parent_i, parent_j, parent_array)
                adj_list[i].append((w, j))
                adj_list[j].append((w, i))
                n_components -= 1

        return Graph(self.__size, self.__labels, adj_list)

    def prim(self) -> 'Graph':
        """
        O(V^2) with adj matrix
        O(E log V) with binary heap
        O(V log V + E) with fibonacci heap """
        pseudocode = False
        if pseudocode:
            raise NotImplementedError

        adj_list: AdjList = [[] for _ in range(self.__size)]

        mst_set = [False] * self.__size
        n_components = self.__size
        pq = PriorityQueue([(0., 0)] + [(float("inf"), v) for v in range(self.__size)])
        keys = [0.] + [float("inf")]*(self.__size - 1)

        while n_components > 1 and not pq.is_empty():
            _, i = pq.dequeue()
            while mst_set[i] and not pq.is_empty():
                _, i = pq.dequeue()

            mst_set[i] = True

            parent: int or None = None
            for w, j in self.adjacent_to(i):
                if not mst_set[j] and w < keys[j]:
                    pq.decrease_key((keys[j], j), (w, j))
                    keys[j] = w
                elif mst_set[j]:
                    parent = j if parent is None or keys[parent] > w else parent

            if parent is not None:
                adj_list[i].append((keys[i], parent))
                adj_list[parent].append((keys[i], i))

        return Graph(self.__size, self.__labels, adj_list)

    @staticmethod
    def _check_frontier(edge: tuple[float, tuple[int, int]], mst_vertices):
        _, (v, u) = edge
        return (v in mst_vertices) ^ (u in mst_vertices)

    @staticmethod
    def _min_edge(edges, mst_vertices):
        n = len(edges)
        i = 0
        while i < n and (not edges[i][1][0] in mst_vertices or edges[i][1][0] in mst_vertices):
            i += 1
        candidate = edges[i]
        while i < n:
            if Graph._check_frontier(edges[i], mst_vertices) and edges[i][0] < candidate[0]:
                candidate = edges[i]
        i += 1
        return candidate

    @staticmethod
    def _find_set(i, representatives: list[int]) -> int:
        p_i = i
        stack = []
        while p_i != representatives[p_i]:
            stack.append(p_i)
            p_i = representatives[p_i]

        for node in stack:
            representatives[node] = p_i

        return p_i

    @staticmethod
    def _join_set(i: int, j: int, representatives: list[int]):
        i = Graph._find_set(i, representatives)
        j = Graph._find_set(j, representatives)
        if i > j:
            representatives[i] = j
        else:
            representatives[j] = i

    def boruvka(self) -> 'Graph':
        """ O(E log V)"""
        adj_list: AdjList = [[] for _ in range(self.__size)]
        parent_array: list[int] = list(range(self.__size))
        n_components = self.__size
        edges = self.edges()
        added_edges = set()

        completed = False
        while not completed and n_components > 1:
            cheapest_edge: list[float or None] = [(None, (None, None)) for _ in range(self.__size)]
            edges_copy = edges.copy()
            edges = []
            for w, (u, v) in edges_copy:
                comp_u = Graph._find_set(u, parent_array)
                comp_v = Graph._find_set(v, parent_array)
                if comp_u != comp_v:
                    if cheapest_edge[comp_u][0] is None or cheapest_edge[comp_u][0] > w:
                        cheapest_edge[comp_u] = (w, (u, v))
                    if cheapest_edge[comp_v][0] is None or cheapest_edge[comp_v][0] > w:
                        cheapest_edge[comp_v] = (w, (u, v))

                    edges.append((w, (u, v)))

            completed = True
            for w, (i, j) in cheapest_edge:
                if w is not None and (i, j) not in added_edges:
                    adj_list[i].append((w, j))
                    adj_list[j].append((w, i))
                    Graph._join_set(i, j, parent_array)
                    n_components -= 1
                    added_edges.add((i, j))
                    completed = False

        return Graph(self.__size, self.__labels, adj_list)

    def topological_sort(self):
        """ Based on dfs -> O(m)"""
        def visit(_v: int):
            if permanent_marked[_v]:
                return
            elif temp_marked[_v]:
                raise Exception("not a DAG")

            temp_marked[_v] = True
            for _, u in self.adjacent_to(_v):
                visit(u)
            temp_marked[_v] = False
            permanent_marked[_v] = True
            solution.append(_v)

        solution = []
        permanent_marked = [False] * self.__size
        temp_marked = [False] * self.__size

        for v in range(self.__size):
            if not permanent_marked[v]:
                visit(v)

        return list(reversed(solution))

    def dfs(self, start: int, allow_forest=True) -> list[int]:
        """ O (E + V) """
        path = []
        visited = [False]*self.__size
        stack = [start]

        while stack:
            i = stack.pop()
            if not visited[i]:
                visited[i] = True
                path.append(i)
                for _, j in self.adjacent_to(i):
                    if not visited[j]:
                        stack.append(j)

        if len(path) < self.__size and allow_forest:
            for v in range(self.__size):
                if not visited[v]:
                    path.extend(self.dfs(v))

        return path

    def recursive_dfs(self, start, path=None, visited=None) -> list[int]:
        """ O(E + V) """
        path = [] if path is None else path
        visited = [False] * self.__size if visited is None else visited

        visited[start] = True
        path.append(start)
        for _, v in self.adjacent_to(start):
            if not visited[v]:
                self.recursive_dfs(v, path, visited)

        return path

    def bfs(self, start):
        """ O(E + V) """
        path = []
        visited = [False] * self.__size
        queue = deque()

        visited[start] = True
        queue.append(start)

        while queue:
            v = queue.popleft()
            path.append(v)

            for _, neighbour in self.adjacent_to(v):
                if not visited[neighbour]:
                    queue.append(neighbour)
                    visited[neighbour] = True

        if len(path) < self.__size:
            for v in range(self.__size):
                if not visited[v]:
                    path.extend(self.bfs(v))

        return path

    def dijkstra(self, source: int, target=None, distance: list[float] = None) -> tuple[list[float], list[int or None]]:
        """
        O(V^2) with adj_matrix
        O(E log V) with adj_list and binary heap
        O(V log V) with Fibonacci heap
        """
        distance = [float("inf")] * self.__size if distance is None else distance
        distance[source] = 0
        prev: list[int or None] = [None] * self.__size
        pq = PriorityQueue([(distance[i], i) for i in range(self.__size)])

        while not pq.is_empty():
            d_u, u = pq.dequeue()
            for w, v in self.adjacent_to(u):
                alt = d_u + w
                if alt < distance[v]:
                    prev[v] = u
                    pq.decrease_key((distance[v], v), (alt, v))
                    distance[v] = alt

            if u == target:
                break

        return distance, prev

    def bellman_ford(self, source: int) -> tuple[list[float], list[int or None]]:
        """ O(EV) """
        distance = [float("inf")] * self.__size
        distance[source] = 0
        prev: list[int or None] = [None] * self.__size

        for _ in range(1, self.__size):
            for w, (u, v) in self.edges():
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
                    prev[v] = u

        for w, (u, v) in self.edges():
            if distance[u] + w < distance[v]:
                raise Exception("Graph contains a negative-weight cycle")

        return distance, prev

    def johnson(self) -> Matrix:
        """ O(V^2 log V + EV) if dijkstra is implemented with a fibonacci heap"""
        adj_list_prime = deepcopy(self.__adj_list) + [[(0, i) for i in range(self.__size)]]

        g_prime = Graph(self.__size + 1, adj_list=adj_list_prime)

        h, _ = g_prime.bellman_ford(self.__size)  # O(EV)

        rw_adj_list = [[] for _ in range(self.__size)]  # rw means reweighted
        for i, L in enumerate(self.__adj_list):
            for w, j in L:
                rw_adj_list[i].append((w + h[i] - h[j], j))

        rw_graph = Graph(self.__size, adj_list=rw_adj_list)

        distances = []
        for v in range(self.__size):  # O(V^2 log V)
            distances.append(rw_graph.dijkstra(v)[0])

        return np.array(distances)

    def is_eulerian(self) -> bool:
        for value in self.__adj_list:
            if len(value) % 2 != 0:
                return False
        return True

    def is_semi_eulerian(self) -> bool:
        odds = 0
        for value in self.__adj_list:
            if len(value) % 2 != 0:
                odds += 1
            if odds > 2:
                return False

        return True

    def hierholzer(self):
        """

        :return: a eulerian trail if the graph is eulerian
        """
        if not self.is_eulerian():
            raise ValueError("the graph is not eulerian")

        edges = self.edges()

        start = 0
        trail = [start]

        degree = []
        dq_adj_list = []
        for value in self.__adj_list:
            dq_adj_list.append(deque(value.copy()))
            degree.append(len(value))

        done = False
        while not done:
            last = start
            w, current = dq_adj_list[start].pop()
            degree[last] -= 1
            degree[current] -= 1
            trail.append(current)
            while current != start:
                last = current
                w, current = dq_adj_list[last].pop()
                degree[last] -= 1
                degree[current] -= 1
                trail.append(current)

            done = True
            for i, d in enumerate(degree):
                if d:
                    done = False
                    start = i

    def directed_hierholzer(self):

        def dfs(v=0):
            while adj_list_copy[v]:
                dfs(adj_list_copy[v].pop()[1])
            res.appendleft(v)

        res = deque()
        adj_list_copy = self.__adj_list.copy()
        dfs()

        return list(res)

    @staticmethod
    def create_graph() -> 'Graph':
        n = int(input("Enter the number of vertex: "))
        graph = Graph(n)
        done = False
        while not done:
            action = int(input("Type the number of the action: "
                               "(0=exit, 1=add_edge, 2=add_arc,"
                               "3=add_weighted_edge, 4=add_weighted_arc) "))
            if action == 0:
                done = True
            elif action == 1:
                i = int(input("vertex 1: "))
                j = int(input("vertex 2: "))
                graph.add_edge(i, j)
            elif action == 2:
                i = int(input("initial vertex: "))
                j = int(input("final vertex: "))
                graph.add_edge(i, j)
            elif action == 3:
                i = int(input("vertex 1: "))
                j = int(input("vertex 2: "))
                w = float(input("weight: "))
                graph.add_edge(i, j, w)
            elif action == 4:
                i = int(input("initial vertex: "))
                j = int(input("final vertex: "))
                w = float(input("weight: "))
                graph.add_edge(i, j, w)

        return graph

    def set_labels(self, dtype: type = str):
        for i in range(self.__size):
            label = dtype(input(f"Label for vertex {i}: "))
            self.set_label(i, label)

    @staticmethod
    def complete_graph(n: int, labels=None, adj_matrix=False, rand_weights=False, w_min=1, w_max=10) -> 'Graph':

        if not rand_weights:
            adj_list = [[(1, j) for j in range(n) if j != i] for i in range(n)]
        else:
            possible_weights = range(w_min, w_max+1)
            adj_list: AdjList = [[] for _ in range(n)]
            edges = combinations(range(n), 2)
            for v1, v2 in edges:
                w = choice(possible_weights)
                adj_list[v1].append((w, v2))
                adj_list[v2].append((w, v1))

        return Graph(n, labels, adj_list, adj_matrix)

    @staticmethod
    def sample_graphs(x: int = 1) -> 'Graph':
        """
        1 -> n = 8 Image: https://bit.ly/3cbSs9q
        2 -> n = 9
        3 -> n = 9
        4 -> n = 6 (DAG)
        5 -> n = 5 eulerian digraph
        """
        if x == 1:
            _g1 = Graph(8)

            _g1.add_edge(0, 2, 5)
            _g1.add_edge(0, 1, 3)
            _g1.add_edge(1, 3, 4)
            _g1.add_edge(2, 3, 12)
            _g1.add_edge(7, 3, 8)
            _g1.add_edge(3, 4, 9)
            _g1.add_edge(6, 7, 20)
            _g1.add_edge(4, 6, 5)
            _g1.add_edge(4, 7)
            _g1.add_edge(5, 4, 4)
            _g1.add_edge(5, 6, 6)

            return _g1

        elif x == 2:
            adj_matrix = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
                          [4, 0, 8, 0, 0, 0, 0, 11, 0],
                          [0, 8, 0, 7, 0, 4, 0, 0, 2],
                          [0, 0, 7, 0, 9, 14, 0, 0, 0],
                          [0, 0, 0, 9, 0, 10, 0, 0, 0],
                          [0, 0, 4, 14, 10, 0, 2, 0, 0],
                          [0, 0, 0, 0, 0, 2, 0, 1, 6],
                          [8, 11, 0, 0, 0, 0, 1, 0, 7],
                          [0, 0, 2, 0, 0, 0, 6, 7, 0]]

            return Graph(9, adj_list=Graph.create_adj_list(adj_matrix), adj_matrix=adj_matrix)

        elif x == 3:
            _g3 = Graph(9)

            _g3.add_edge(0, 1, 4)
            _g3.add_edge(0, 7, 8)
            _g3.add_edge(1, 2, 8)
            _g3.add_edge(1, 7, 11)
            _g3.add_edge(2, 3, 7)
            _g3.add_edge(2, 8, 2)
            _g3.add_edge(2, 5, 4)
            _g3.add_edge(3, 4, 9)
            _g3.add_edge(3, 5, 14)
            _g3.add_edge(4, 5, 10)
            _g3.add_edge(5, 6, 2)
            _g3.add_edge(6, 7, 1)
            _g3.add_edge(6, 8, 6)
            _g3.add_edge(7, 8, 7)

            return _g3

        elif x == 4:
            _g4 = Graph(6)

            _g4.add_arc(5, 2)
            _g4.add_arc(5, 0)
            _g4.add_arc(4, 0)
            _g4.add_arc(4, 1)
            _g4.add_arc(2, 3)
            _g4.add_arc(3, 1)

            return _g4
        elif x == 5:
            _g5 = Graph(5)

            _g5.add_arc(0, 2)
            _g5.add_arc(0, 1)
            _g5.add_arc(1, 3)
            _g5.add_arc(2, 3)
            _g5.add_arc(3, 0)
            _g5.add_arc(3, 4)
            _g5.add_arc(4, 0)

            return _g5

        else:
            raise IndexError()

    def __str__(self):

        if self.__adj_matrix is not None:
            return "Graph(\nlabels: " + str(self.__labels) + ",\n" \
                    + "adjacency matrix:\n" + str(self.adj_matrix) + ")" \
                    + "\nadjacency_list: " + str(self.__adj_list)
        else:
            return "Graph(\nlabels: " + str(self.__labels) + ",\n" \
                   + "adjacency_list: " + str(self.__adj_list)


def main():
    g = Graph.sample_graphs(5)
    print(g.is_eulerian())
    print(g.directed_hierholzer())


def main2():
    a = [[(10, 1), (5, 2), (9, 3), (2, 4)],
         [(10, 0), (10, 2), (7, 3), (9, 4)],
         [(5, 0), (10, 1), (5, 3), (7, 4)],
         [(9, 0), (7, 1), (5, 2), (1, 4)],
         [(2, 0), (9, 1), (7, 2), (1, 3)]]

    k_5 = Graph(5, adj_list=a)
    print(k_5.floyd())
    print(k_5.johnson())


if __name__ == "__main__":
    main()
