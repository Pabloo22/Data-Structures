

class CircularQueue:

    __items: list
    __max_size: int
    __size: int
    __front: int = 1
    __rear: int = 0

    def __init__(self, max_size: int):
        self.__items = [None for _ in range(max_size)]
        self.__max_size = max_size
        self.__size = 0

    def is_empty(self) -> bool:
        """

        :return: True if the queue is empty and False if it is not empty
        """
        return self.__size == 0

    def is_full(self) -> bool:
        """

        :return: True if the queue is full and False if it is not
        """
        return self.__size == self.__max_size

    def dequeue(self):
        """
        Removes the first index of the queue and returns it.

        :return: the last item of the queue
        """
        if self.is_empty():
            raise IndexError("dequeue from empty queue")

        self.__size -= 1
        item = self.__items[self.__front]
        self.__items[self.__front] = None
        self.__front = (self.__front + 1) % self.__max_size

        return item

    def enqueue(self, item):
        """
        Adds a new index in the last position of the queue.

        :param item: the index which will be added
        :return: None
        """
        if self.is_full():
            raise IndexError("enqueue from full queue")

        self.__size += 1
        self.__items[self.__rear] = item
        self.__rear = (self.__rear + 1) % self.__max_size

    def to_list(self) -> list:
        """

        :return: the queue converted to list
        """
        return self.__items

    @property
    def size(self):
        return self.__max_size

    @property
    def num_elements(self):
        return self.__size

    def __str__(self):
        return str(self.__items)