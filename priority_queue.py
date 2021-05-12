from Trees.bynary_heap import BinaryHeap


class PriorityQueue:
    """
    Retrieves open entries in priority order (lowest first).
    The elements in the heap are tuples of the form: (priority number, tie breaker (insert order), data)
    """
    __heap: BinaryHeap
    __insert_order: int

    def __init__(self):
        self.__heap = BinaryHeap()
        self.__insert_order = 0

    def enqueue(self, priority: float, data: any):
        self.__heap.push((priority, self.__insert_order, data))
        self.__insert_order += 1

    def dequeue(self):
        _, _, element = self.__heap.pop()
        return element

    def peek(self):
        _, _, element = self.__heap.peek()
        return element

    @property
    def size(self):
        return self.__heap.size

    def is_empty(self):
        return self.__heap.is_empty()


if __name__ == "__main__":
    a = PriorityQueue()

