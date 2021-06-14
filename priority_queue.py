from bynary_heap import BinaryHeap
from typing import TypeVar

_T = TypeVar("_T")


class PriorityQueue:
    """
    Retrieves open entries in priority order (lowest first).
    The elements in the heap are tuples of the form: (priority number, tie breaker (insert order), data)
    """
    __heap: BinaryHeap

    def __init__(self, values: list[tuple[float or int, _T]] = None):
        values = [] if values is None else values

        if not isinstance(values, list):
            raise ValueError("'values' must be a list")
        for v in values:
            if (not isinstance(v, tuple) or len(v) != 2 and
                    (not isinstance(v[0], float) or not isinstance(v[0], int))):
                raise ValueError("'values' must be a list of tuples with the form " +
                                 "(priority, value)")

        self.__heap = BinaryHeap(values)

    def enqueue(self, priority: float or int, data: _T):
        self.__heap.push((priority, data))

    def dequeue(self):
        return self.__heap.pop()

    def peek(self):
        return self.__heap.peek()

    def decrease_key(self, old: _T, new: _T):
        self.__heap.decrease_key(old, new)

    @property
    def size(self):
        return self.__heap.size

    def is_empty(self):
        return self.__heap.is_empty()

    def __str__(self):
        return str(self.__heap)


if __name__ == "__main__":
    a = PriorityQueue()
