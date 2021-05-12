from weighted_directed_vertex import WDVertex


class Edge:
    __weight: float
    __v1: WDVertex
    __v2: WDVertex
    __label: any

    def __init__(self, v1, v2, weight=1):
        self.__v1 = v1
        self.__v2 = v2
        self.__weight = weight
        self.__label = False

    @property
    def weight(self):
        return self.__weight

    @property
    def v1(self):
        return self.__v1

    @property
    def v2(self):
        return self.__v2

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label=True):
        self.__label = label

    def __str__(self):
        return str(self.__v1) + "-" + str(self.__weight) + "-" + str(self.__v2)


class Arc(Edge):

    def __init__(self, v1, v2, weight=1):
        super().__init__(v1, v2, weight)

    def __str__(self):
        return str(self.__v1) + "-" + str(self.__weight) + "->" + str(self.__v2)