

class WDVertex:
    __in_adj_vertices: list  # contains other WDVertex objects
    __out_adj_vertices: list  # contains other WDVertex objects
    __adj_vertices: list
    __name: str
    __weight: float
    __label: any

    def __init__(self, name="", weight=1):
        self.__in_adj_vertices = []
        self.__out_adj_vertices = []
        self.__adj_vertices = []
        self.__name = name
        self.__weight = weight

    def set_adj_vertices(self, vertices: object or list):
        if isinstance(vertices, WDVertex):
            self.__in_adj_vertices.append(vertices)
            self.__out_adj_vertices.append(vertices)
        else:
            self.__in_adj_vertices.extend(vertices)
            self.__out_adj_vertices.extend(vertices)

    def degree(self):
        if len(self.__in_adj_vertices) == len(self.__out_adj_vertices):
            return len(self.__in_adj_vertices)
        else:
            raise SyntaxError("out_degree and in_degree are different")

    def out_degree(self):
        return len(self.__out_adj_vertices)

    def in_degree(self):
        return len(self.__in_adj_vertices)

    @property
    def weight(self):
        return self.__weight

    @property
    def in_adj_vertices(self):
        return self.__in_adj_vertices

    @property
    def out_adj_vertices(self):
        return self.__out_adj_vertices

    @property
    def adj_vertices(self):
        return self.__adj_vertices

    def add_adj_vertex(self, v):
        self.__adj_vertices.append(v)

    def remove_adj_vertex(self, v):
        self.__adj_vertices.remove(v)

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label=True):
        self.__label = label

    def __str__(self):
        return self.__name

    def __hash__(self):
        return hash(self.__name)


if __name__ == "__main__":
    from edges import Edge

    a = WDVertex("a")
    b = WDVertex("b")
    a.set_adj_vertices(b)
    b.set_adj_vertices(a)
    ab = Edge(a, b)
    print(a.degree())
    print(b.adj_vertices())
    print(ab)
    print()
