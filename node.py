

class Node:

    def __init__(self, value):
        self.__value = value
        self.__next = None
        self.__previous = None
        self.__distance_to_next = None

    def __str__(self):
        return 'Node(' + str(self.__value) + ')'

    def __del__(self):
        pass

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @property
    def next(self):
        return self.__next

    @next.setter
    def next(self, nxt):
        self.__next = nxt

    @property
    def previous(self):
        return self.__previous

    @previous.setter
    def previous(self, previous):
        self.__previous = previous

    @property
    def distance_to_next(self):
        return self.__distance_to_next

    @distance_to_next.setter
    def distance_to_next(self, new_distance):
        if new_distance < 1 or not isinstance(new_distance, int):
            raise ValueError("distance between two nodes must be a positive integer")

        self.__distance_to_next = new_distance
