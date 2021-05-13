import numpy as np
from copy import deepcopy
from typing import TypeVar

Matrix = TypeVar('Matrix', list[list[float]], np.ndarray)


class Graph:

    __size: int
    __adj_list: list[list[tuple]]
    __adj_matrix: None or Matrix
    __labels: list

    def __init__(self, n_nodes: int = 0, labels: list = None):
        if isinstance(labels, list):
            if len(labels) != n_nodes:
                raise ValueError("len(labels) != n_nodes")

        self.__size = n_nodes
        self.__adj_list = [[] for _ in range(n_nodes)]
        self.__labels = [None] * n_nodes if labels is None else labels
        self.__adj_matrix = None

    @property
    def size(self):
        return self.__size

    @property
    def adj_matrix(self):
        if self.__adj_matrix is None:
            adj_matrix = np.zeros((self.__size, self.__size), dtype=float)
            for i, value in enumerate(self.__adj_list):
                for j, w in value:
                    adj_matrix[i][j] = w

            self.__adj_matrix = adj_matrix

        return self.__adj_matrix
        
    @property
    def labels(self):
        return self.__labels

    def add_edge(self, i: int, j: int, weight: float = 1):
        if self.__adj_matrix is not None:
            self.adj_matrix[i][j] = weight
            self.adj_matrix[j][i] = weight

    def add_arc(self, i: int, j: int, weight: float = 1):
        if self.__adj_matrix is not None:
            self.adj_matrix[i][j] = weight

    def set_label(self, i: int, value: any):
        if self.__adj_matrix is not None:
            self.__labels[i] = value

    def get_label(self, i: int) -> any:
        return self.__labels[i]

    def get_adjacent_to(self, i: int) -> list[tuple[int, float]]:
        return self.__adj_list[i]

    def warshall(self) -> Matrix:
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

    def get_distances(self) -> Matrix:

        n = len(self.adj_matrix)
        dist_matrix = deepcopy(self.adj_matrix)
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    if dist_matrix[i][j] or (dist_matrix[i][k] and dist_matrix[k][j]):
                        if not dist_matrix[i][j]:
                            dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                        elif not (dist_matrix[i][k] and dist_matrix[k][j]):
                            dist_matrix[i][j] = dist_matrix[i][j]
                        else:
                            dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
        return dist_matrix

    def are_connected(self, i, j) -> bool:
        pass

    @staticmethod
    def create_graph() -> 'Graph':
        def is_valid_vertex(i, n):
            return 0 <= i <= n-1

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
                while not is_valid_vertex(i, n):
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

    def __str__(self):
        return "Graph(\nlabels: " + str(self.__labels) + ",\n" \
                + "adjacency matrix:\n" + str(self.__adj_matrix) + ")"


if __name__ == "__main__":
    G = Graph(4)
    G.add_arc(0, 2, 4)
    G.add_arc(1, 0, 1)
    G.add_arc(1, 3, 3)
    G.add_arc(3, 1, 5)
    print(G)
    print(G.warshall())
    print(G.get_distances())
