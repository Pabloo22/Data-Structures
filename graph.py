from typing import List, Tuple
import numpy as np
from weighted_directed_vertex import WDVertex
from edges import Edge


class Graph:

    __vertices_str: List[str]
    __vertices: List[WDVertex]
    __edges: List[Edge]
    __adj_list: List[int or str]  # keys: vertices_, values: set of adjacent vertices_
    __length: int

    def __init__(self, vertices: List[str] = None,
                 adj_list: List[List[str] or List[Tuple[str, float]]] = None):

        self.__vertices_str = vertices if vertices is not None else []
        self._set_vertices()
        self.__adj_list = adj_list if adj_list is not None else []
        self.__set_edges()

    def _set_vertices(self):
        self.__vertices = []
        for v in self.__vertices_str:
            if isinstance(v, list) or isinstance(v, tuple):
                self.__vertices.append(WDVertex(*v))
            else:
                self.__vertices.append(WDVertex(str(v)))

    def __set_edges(self):
        self.__edges = []
        for i, list_ in enumerate(self.__adj_list):
            for v in list_:
                v1 = str(self.__vertices[i])
                if isinstance(v, list) or isinstance(v, tuple):
                    self.add_edge(v1, v[0], v[1])
                else:
                    v2 = self.__vertices[self.get_vertex_index(v)]
                    self.__edges.append(Edge(v1, v2))
                    self.add_edge(v1, v)

    def get_vertex_index(self, vertex):
        return self.__vertices_str.index(str(vertex))

    def get_vertex_from_str(self, name: str or int) -> WDVertex:
        return self.__vertices[self.get_vertex_index(name)]

    def add_vertex(self, vertex: any, weight=1):
        if not isinstance(vertex, WDVertex):
            vertex = WDVertex(str(vertex), weight)

        self.__vertices.append(vertex)
        self.__vertices_str.append(str(vertex))
        self.__adj_list.append([])

    def add_vertices(self, *vertices: WDVertex or list):
        for v in vertices:
            if isinstance(v, list) or isinstance(v, tuple):
                self.add_vertex(*v)
            else:
                self.add_vertex(v)

    def add_edge(self, v1: str, v2: str, weight=1):
        i = self.get_vertex_index(v1)
        j = self.get_vertex_index(v2)
        self.__adj_list[i].append((str(v2), weight))
        self.__adj_list[j].append((str(v1), weight))

        v1 = self.get_vertex_from_str(v1)
        v2 = self.get_vertex_from_str(v2)
        v1.add_adj_vertex(v2)
        v2.add_adj_vertex(v2)

        self.__edges.append(Edge(v1, v2, weight))

    def add_edges(self, *edges: Edge):
        for e in edges:
            self.add_edge(e.v1, e.v2, e.weight)

    @property
    def vertices_str(self):
        return self.__vertices_str

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices):

        def change_name(old_list: list):
            new_list = []
            for element in old_list:
                if isinstance(element, tuple) or isinstance(element, list):
                    new_list.append((str(new_names[element[0]]), element[1]))
                else:
                    new_list.append(str(new_names[element[0]]))

            return new_list

        vertices = list(map(str, vertices))
        new_names = {old_name: new_name for old_name, new_name in zip(self.__vertices_str, vertices)}
        new_adj_list = list(map(change_name, self.__adj_list))
        self.__init__(vertices, new_adj_list)

    @property
    def edges(self):
        return self.__edges

    def print_edges(self):
        for edge in self.__edges:
            print(edge)

    def get_adj_matrix(self):
        adj_matrix = np.zeros((len(self.__vertices), len(self.__vertices)))
        for edge in self.__edges:
            adj_matrix[self.__vertices.index(edge.v1)][self.__vertices.index(edge.v2)] = edge.weight
            adj_matrix[self.__vertices.index(edge.v2)][self.__vertices.index(edge.v1)] = edge.weight

        return adj_matrix

    @staticmethod
    def build_adj_list(adj_matrix, vertices):
        _adj_list = [[] for _ in range(len(vertices))]
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] != 0:
                    _adj_list[i].append((vertices[j], adj_matrix[i][j]))

        return _adj_list

    @staticmethod
    def check_degree_sequence(sequence: list):
        """
        This algorithm is based on the Havel-Hakimi theorem.
        Returns True if the degree sequence is valid and returns False
        if it isn't.
        It is a static method
        """

        while True:
            sequence.sort(reverse=True)  # O(n log n)
            first = sequence.pop(0)

            for i in range(len(sequence[:first])):
                sequence[i] -= 1

            if -1 in sequence:
                return False
            elif max(sequence) == 0:
                return True

    def breadth_first_search(self, start=None):
        """ In development """
        first_vertex_str = str(start) if start is not None else self.__vertices_str[0]
        first_vertex = self.get_vertex_from_str(first_vertex_str)

        unexplored_vertices = [v for v in self.__vertices]
        unexplored_vertices.remove(first_vertex)
        tree = Graph([str(first_vertex)])
        tree.add_vertex(first_vertex)
        while unexplored_vertices:
            unexplored_vertices.remove(first_vertex)
            tree.add_vertex(first_vertex)

    def __len__(self):
        return len(self.__vertices)

    def __str__(self):
        return str(self.__adj_list)


if __name__ == "__main__":
    vertices__ = ["a", "b", "c", "d"]
    adj_list__ = [[("a", 3), ("c", 2)], [("c", 2)], [("d", -2), ("a", 2)], [("c", -2)]]
    my_graph = Graph(vertices__, adj_list__)
    my_graph.vertices = [1, 2, 3, 4]
    a = WDVertex("a")
    print(a)
