from node import Node


class LinkedList:

    def __init__(self):
        self.__head = None
        self.__size = 0

    def __str__(self):
        return 'Head: [' + str(self.__head) + ']'

    def is_empty(self):
        return self.__head is None

    def size(self):
        return self.__size

    def first(self):
        if self.is_empty():
            raise IndexError('NO first item in empty list')

        return self.__head.value

    def last(self):
        if self.is_empty():
            raise IndexError('NO last item in empty list')

        current = self.__head
        p_next = self.__head

        while not (p_next is None):
            current = p_next
            p_next = p_next.next

        return current.value

    def insert_rear(self, ele):
        temp = Node(ele)
        self.__size += 1
        if self.__head is None:
            self.__head = temp
        else:
            p_next = self.__head
            while not (p_next.next is None):
                p_next = p_next.next
            p_next.next = temp

    def insert_top(self, ele):
        temp = Node(ele)
        self.__size += 1
        temp.next = self.__head
        self.__head = temp

    def insert_at(self, pos, ele):
        if pos > self.__size:
            raise IndexError('pos higher than size of the list')

        self.__size += 1
        if pos == 0:
            self.insert_top(ele)
        else:
            if pos == self.__size:
                self.insert_rear(ele)
            else:
                temp = Node(ele)
                current = None
                successor = self.__head
                for i in range(pos):
                    current = successor
                    successor = successor.next
                current.next = temp
                temp.next = successor

    def remove(self, ele):
        if not self.is_in(ele):
            raise IndexError('the item is not in the list')

        self.__size -= 1
        if self.__head.value == ele:
            self.__head = self.__head.next
        else:
            current = None
            successor = self.__head
            while successor.value != ele and successor.next is not None:
                current = successor
                successor = successor.next
            current.next = successor.next

    def is_in(self, ele):
        if self.is_empty():
            return False

        temp = self.__head
        while temp.value != ele and temp.next is not None:
            temp = temp.next

        return temp.value == ele

