from typing import TypeVar

_T = TypeVar('_T')


class BinaryHeap:
    """
    A binary heap is defined as a binary tree with two additional constraints:
      - Shape property: a binary heap is a complete binary tree; that is, all levels of the tree, except possibly the
        last one (deepest) are fully filled, and, if the last level of the tree is not complete, the nodes of that level
        are filled from left to right.
      - Heap property: the key stored in each node is either greater than or equal to (≥) or less than or equal to (≤)
        the keys in the node's children, according to some total order.

    (https://en.wikipedia.org/wiki/Binary_heap)
    """

    __values: list[_T]

    def __init__(self, values: list[_T] = None):
        self.__values = [] if values is None else self.heapify(values)
        self.__size = len(self.__values)

    def push(self, item: _T):
        """
        Adding a new node to the heap
        :param item: the item to be added
        """
        self.__values.append(item)

        child_index = len(self.__values) - 1
        self.__bubble_up(child_index)

    def pop(self) -> any:
        """
        Returns the node of minimum value from a min heap after removing it from the heap
        :return: the node of minimum value
        """
        if self.is_empty():
            raise IndexError("pop from empty queue")
        removed = self.__values[0]
        if self.size > 1:
            self.__values[0] = self.__values.pop()
            self.__bubble_down(0)
        else:
            self.__values.pop()

        return removed

    def peek(self):
        """
        Returns the node of minimum value from a min heap without removing it from the heap
        :return: the node of minimum value
        """
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.__values[0]

    def replace(self, item: _T) -> _T:
        """
        Pop root and push a new item. More efficient than pop followed by push, since only need to balance once,
        not twice.
        :return: the node of minimum value
        """
        if self.is_empty():
            raise IndexError("replace from empty queue")
        removed = self.__values[0]
        self.__values[0] = item
        self.__bubble_down(0)

        return removed


    @staticmethod
    def heapify(x: list[_T]) -> 'BinaryHeap':
        """
        Create a heap out of given list of sequence. It uses Floyd Algorithm.
        :param x: A list of _T items.
        :return: the heap with that items
        """

    @staticmethod
    def merge(heap1: 'BinaryHeap', heap2: 'BinaryHeap') -> 'BinaryHeap':
        """
        Joining two heaps to form a valid new heap containing all the sequence of both, preserving the original heaps.
        :param heap1: the first heap to join
        :param heap2: the second heap to join
        :return: a valid new heap containing all the sequence of both heaps
        """

    @property
    def values(self):
        return self.__values

    @property
    def size(self) -> int:
        """
        :return: the number of items in the heap
        """
        return len(self.__values)

    def is_empty(self) -> bool:
        """
        :return: True if the heap is empty, False otherwise
        """
        return not self.__values

    def __bubble_up(self, child: int):
        """
        Move a node up in the tree, as long as needed. Used to restore heap condition after insertion.
        """
        a = self.__parent(child)
        b = self.__values[child]
        while self.__parent(child) > self.__values[child]:
            parent = (child-1) // 2
            self.__swap(parent, child)
            child = parent

    def __bubble_down(self, parent: int):
        """
        Move a node down in the tree, similar to bubble-up. Used to restore heap condition after
        deletion or replace.
        """
        min_child = min(self.__left_child(parent), self.__right_child(parent))
        while self.__values[parent] > min_child:
            child = 2*parent + 1 if self.__left_child(parent) < self.__right_child(parent) else 2*parent + 2
            self.__swap(parent, child)
            parent = child
            min_child = min(self.__left_child(parent), self.__right_child(parent))

    def __swap(self, node1: int, node2: int):
        """
        Exchange the position of the two given nodes
        :param node1: index of the first node
        :param node2: index of the second node
        :return:
        """
        self.__values[node1], self.__values[node2] = self.__values[node2], self.__values[node1]

    def __parent(self, child: int) -> any:
        """return the index of the parent of the node given by his index"""
        return float("-inf") if child == 0 else self.__values[(child-1) // 2]

    def __left_child(self, i: int) -> _T:
        """return the index of the right child of the node given by his index"""
        j = 2*i + 1
        return float("inf") if j >= len(self.__values) else self.__values[j]

    def __right_child(self, i: int) -> _T:
        """return the index of the right child of the node given by his index"""
        j = 2*i + 2
        return float("inf") if j >= len(self.__values) else self.__values[j]


if __name__ == "__main__":
    bh = BinaryHeap()
    bh.push(10)
    print(bh.pop())
    print(bh.values)
    bh.push(4)
    bh.push(15)
    print(bh.pop())
    bh.push(20)
    bh.push(0)
    bh.push(30)
    print(bh.values)
    print(bh.pop())
    print(bh.pop())
    bh.push(2)
    bh.push(4)
    bh.push(-1)
    bh.push(-3)
    print(bh.values)
