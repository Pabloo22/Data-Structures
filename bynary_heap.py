from typing import TypeVar
from math import log2

_T = TypeVar('_T')  # This indicates that the items can be of any type and that all of them should be of the same type


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
        self.__values = [] if values is None else values
        self.__heapify()

    def push(self, item: _T):
        """
        Adding a new node to the heap
        :param item: the item to be added
        """
        self.__values.append(item)

        child_index = len(self.__values) - 1
        self.__bubble_up(child_index)

    def pop(self) -> _T:
        """
        Returns the node of minimum value from a min heap after removing it from the heap
        :return: the node of minimum value
        """
        if self.is_empty():
            raise IndexError("pop from empty queue")
        removed = self.__values[0]
        if self.size() > 1:
            self.__values[0] = self.__values.pop()
            self.__bubble_down(0)
        else:
            self.__values.pop()

        return removed

    def peek(self) -> _T:
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

    def __heapify(self):
        for i in reversed(range(self.size() // 2)):
            self.__bubble_down(i)

    @staticmethod
    def merge(heap1: 'BinaryHeap', heap2: 'BinaryHeap') -> 'BinaryHeap':
        """
        Joining two heaps to form a valid new heap containing all the sequence of both, preserving the original heaps.
        :param heap1: the first heap to join
        :param heap2: the second heap to join
        :return: a valid new heap containing all the sequence of both heaps
        """
        return BinaryHeap(heap1.values.extend(heap2.values))

    @property
    def values(self):
        return self.__values

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
        parent_value = self.__parent(child)
        while parent_value is not None and parent_value > self.__values[child]:
            parent = (child-1) // 2
            self.__swap(parent, child)
            child = parent
            parent_value = self.__parent(child)

    def __bubble_down(self, parent: int):
        """
        Move a node down in the tree, similar to bubble-up. Used to restore heap condition after
        deletion or replace.
        """
        min_child = self.__get_min_child(parent)
        while min_child is not None and self.__values[parent] > min_child:
            child_index = 2*parent + 1 if min_child == self.__left_child(parent) else 2*parent + 2
            self.__swap(parent, child_index)
            parent = child_index
            min_child = self.__get_min_child(parent)

    def __get_min_child(self, parent: int):
        left_child_value = self.__left_child(parent)
        right_child_value = self.__right_child(parent)
        if left_child_value is not None or right_child_value is not None:
            if left_child_value is None:
                return right_child_value
            elif right_child_value is None:
                return left_child_value
            else:
                return min(self.__left_child(parent), self.__right_child(parent))

    def __swap(self, node1: int, node2: int):
        """
        Exchange the position of the two given nodes
        :param node1: index of the first node
        :param node2: index of the second node
        """
        self.__values[node1], self.__values[node2] = (self.__values[node2],
                                                      self.__values[node1])

    def __parent(self, child: int) -> _T or None:
        """return the index of the parent of the node given by his index.
        Returns -infinite if the given node is the root"""
        i_parent = (child-1) // 2
        return None if child == 0 else self.__values[i_parent]

    def __left_child(self, i: int) -> _T or None:
        """return the index of the right child of the node given by his index.
        Returns infinite if no left child."""
        i_left_child = 2*i + 1
        return (None if i_left_child >= len(self.__values) else
                self.__values[i_left_child])

    def __right_child(self, i: int) -> _T or None:
        """return the value of the right child of the node given by his index.
        Returns infinite if no right child. """
        i_right_child = 2*i + 2
        return None if i_right_child >= len(self.__values) else self.__values[i_right_child]

    def print_tree(self):
        ts = int(log2(self.size()))
        tree = "\t" * ts
        ts -= 1
        mod = 1
        counter = -1
        for v in self.__values:
            tree += " "*(ts+2)**2 + f" {v} " + " "*(ts+2)**2
            counter += 1
            if counter % mod == 0:
                counter = 0
                tree += "\n" + "\t"*ts
                ts -= 1
                mod *= 2
        print(tree)

    def __str__(self):
        return f"Heap{self.__values}"

    def __del__(self):
        pass


if __name__ == "__main__":
    import numpy as np

    bh = BinaryHeap()
    bh.push(10)
    bh.push(4)
    bh.push(15)
    bh.push(20)
    bh.push(0)
    bh.push(30)
    bh.push(2)
    bh.push(4)
    bh.push(-1)
    bh.push(-3)

    bh2 = BinaryHeap(list(np.random.randint(low=0, high=100, size=15)))
    bh2.print_tree()
    print(bh2)
