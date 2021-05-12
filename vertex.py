

class Vertex:

    __adj_vertices: list
    __name: str

    def __init__(self, name: str or int):
        self.__name = name
        self.__adj_vertices = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def degree(self):
        return len(self.__adj_vertices)

    @property
    def adj_vertices(self):
        return self.__adj_vertices

    @adj_vertices.setter
    def adj_vertices(self, vertices: list['Vertex']):
        self.__adj_vertices = vertices

    def add_adj_vertex(self, v):
        self.__adj_vertices.append(v)

    def remove_adj_vertex(self, v):
        """O(N)"""
        self.__adj_vertices.remove(v)

    def __str__(self):
        return self.__name

    def __hash__(self):
        return hash(self.__name)

    def __del__(self):
        pass
