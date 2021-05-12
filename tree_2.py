from vertex import Vertex


class Tree:

    __labels: list
    __adj_list: list[list[int] or None]
    __vertices: list[Vertex or None]

    def __init__(self,
                 adj_list: list[list[int]],
                 labels: list = None):

        self.__adj_list = adj_list
        self.__labels = [None] * len(adj_list) if labels is None else labels
        self.__vertices = [Vertex(name) for name in self.__labels]

        for i, neighbours in enumerate(self.__adj_list):
            for j in neighbours:
                self.__vertices[i].add_adj_vertex(self.__vertices[j])

    @property
    def vertices(self):
        return self.__vertices

    @property
    def adj_list(self):
        return self.__adj_list

    def add_vertex(self, name=None):

        vertex = Vertex(name)
        self.__vertices.append(vertex)
        self.__adj_list.append([])
        self.__labels.append(name)

    def remove_vertex(self, i: int):
        self.__vertices[i] = None
        self.__adj_list[i] = None
        self.__labels[i] = None

    def add_edge(self, i: int, j: int) -> None:

        self.__adj_list[i].append(j)
        self.__adj_list[j].append(i)
        v1 = self.__vertices[i]
        v2 = self.__vertices[j]
        v1.add_adj_vertex(v2)
        v2.add_adj_vertex(v1)

    def __len__(self):
        return len(self.__vertices)

    def __str__(self):
        return str(dict(zip(range(len(self.__adj_list)), self.__adj_list))) + "\nlabels: " + str(self.__labels)

    @staticmethod
    def prufer_code(adj_list: list[list[int]]) -> list[int]:
        """
        :return: the prufer code of the tree
        """
        # Preprocessing:
        parent_array = [-2] * len(adj_list)
        parent_array[len(adj_list)-1] = -1
        nodes_of_the_level = [len(adj_list)-1]
        while nodes_of_the_level:  # O(N + q) = O(N + N-1) = O(N)
            aux = nodes_of_the_level.copy()
            nodes_of_the_level = []
            for i in aux:
                for j in adj_list[i]:
                    if parent_array[j] == -2:
                        parent_array[j] = i
                        nodes_of_the_level.append(j)

        degrees = []
        min_leaf = -1
        for i, neighbours in enumerate(adj_list):
            if len(neighbours) == 1 and min_leaf == -1:
                index = i
                min_leaf = i
            degrees.append(len(neighbours))

        # Algorithm:
        prufer_code = []
        for i in range(len(adj_list)-2):
            parent = parent_array[min_leaf]
            prufer_code.append(parent)
            degrees[parent] -= 1
            if parent < index and degrees[parent] == 1:
                min_leaf = parent
            else:
                for j in range(index+1, len(adj_list)):
                    if degrees[j] == 1:
                        index = j
                        min_leaf = j
                        break

        return prufer_code

    @staticmethod
    def get_tree_from_prufer_code(prufer_code: list[int], names: list = None) -> 'Tree':
        """
        The complexity of this solution is O(n), where n is the number of vertices.
        Args:
            names: a list with the names of the vertices of our tree.
            prufer_code: the Prüfer code

        Returns: the tree associated with the given Prüfer code
        """
        adj_list = [[] for _ in range(len(prufer_code) + 2)]
        tree = Tree(adj_list, names)

        # Preprocessing:
        degrees = [1] * len(adj_list)
        for i in prufer_code:
            degrees[i] += 1

        # Algorithm:
        min_leaf = degrees.index(1)
        for _ in range(len(adj_list) - 2):
            parent = prufer_code[min_leaf]
            tree.add_edge(min_leaf, parent)
            degrees[parent] -= 1

            if parent < index and degrees[parent] == 1:
                min_leaf = parent
            else:
                for i in range(index + 1, len(adj_list)):
                    if degrees[i] == 1:
                        index = i
                        min_leaf = i
                        break

        return tree

