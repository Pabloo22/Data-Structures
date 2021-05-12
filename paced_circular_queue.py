from node import Node


class PacedCircularQueue:

    __head: Node  # It does not contains a value of the queue
    __size: int
    __max_size: int
    __last_added_node: Node
    __last_added_distance: int
    __next_node: Node  # The next node to start counting when we remove

    def __init__(self, max_size: int):
        self.__head = Node(None)
        self.__head.distance_to_next = max_size + 1
        self.__head.next = self.__head
        self.__head.previous = self.__head
        self.__head.index = -1

        self.__size = 0
        self.__max_size = max_size
        self.__last_added_node = self.__head
        self.__last_added_distance = 1
        self.__next_node = self.__head

    def __str__(self):
        return "Head:" + str(self.get_all_values())

    def is_empty(self):
        return self.__size == 0

    def is_full(self):
        return self.__size == self.__max_size

    @property
    def size(self):
        return self.__size

    @property
    def max_size(self):
        return self.__max_size

    @property
    def next_node(self):
        return self.__next_node

    @property
    def last_added_node(self):
        return self.__last_added_node

    def insert(self, ele):
        """
        Adds a new item in the next free space.
        :param ele: the value which will be added
        :return: None
        """
        if self.is_full():
            raise IndexError('cannot insert item to full queue')

        temp = Node(ele)

        if self.is_empty():
            self.__next_node = temp

        last = self.__last_added_node

        while last.distance_to_next == 1:
            last = last.next

        # setting distance
        temp.distance_to_next = last.distance_to_next - self.__last_added_distance
        last.distance_to_next = self.__last_added_distance

        # setting next
        temp.next = last.next
        last.next = temp

        # setting previous
        temp.previous = last
        temp.next.previous = temp

        self.__size += 1
        self.__last_added_node = temp
        self.__last_added_distance = 1

    def remove(self, n: int, pace: int = 1):
        """
        Removes N sequence following a given step (1 by default)
        :param n: number of sequence
        :param pace: number of nodes between two eliminations
        :return: None
        """
        if pace < 1:
            raise ValueError('pace must be a positive integer')

        if self.is_empty():
            raise IndexError('cannot remove items from empty queue')

        if self.__size < n:
            raise ValueError('there is no enough items to be removed')

        for _ in range(n):
            current = self.__next_node
            for _ in range(pace - 1):
                current = current.next

                if current is self.__head:  # the head cannot be removed
                    current = current.next

            # setting the next node to start counting
            if current.next is self.__head and self.__size > 1:
                self.__next_node = self.__head.next
            else:
                self.__next_node = current.next

            # if we delete the last added node we must set a new one.
            if current is self.__last_added_node:
                if current.distance_to_next == 1:
                    self.__last_added_node = current.next
                else:
                    self.__last_added_node = current.previous
                    self.__last_added_distance += current.previous.distance_to_next

            # removing the current node
            current.previous.next = current.next
            current.previous.distance_to_next += current.distance_to_next
            current.next.previous = current.previous

            del current
            self.__size -= 1

    def index(self, node: Node) -> int:
        """
        :param node: the Node object (not the value)
        :return: the index of the node that would have in the isomorphic list
        """
        index = -1
        current = self.__head
        while current is not node:
            index += current.distance_to_next
            current = current.next
            if current == self.__head:
                raise IndexError('the given node is not in the queue')

        return index

    def get_all_values(self) -> list:
        """
        :return: the isomorphic list to the queue
        """
        current = self.__head
        values = [None] * (self.__head.distance_to_next - 1)

        while current.next is not self.__head:
            current = current.next
            values.append(current.value)
            values += [None] * (current.distance_to_next - 1)

        return values

    def __iter__(self):
        return iter(self.get_all_values())
