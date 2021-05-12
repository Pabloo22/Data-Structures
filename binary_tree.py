

class BinaryTree:
    def __init__(self, ele=None):
        self.__parent = None
        self.__element = ele
        self.__left_child = None
        self.__right_child = None

    @property
    def element(self):
        return self.__element

    @element.setter
    def element(self, ele):
        self.__element = ele

    def parent(self):
        return self.__parent

    def is_root(self):
        return self.parent() is None

    @property
    def left_child(self):
        return self.__left_child

    @property
    def right_child(self):
        return self.__right_child

    def append_left_child(self, child):
        child.__parent = self
        if self.__left_child is None:
            self.__left_child = child
        else:
            raise BaseException

    def append_right_child(self, child):
        child.__parent = self
        if self.__right_child is None:
            self.__right_child = child
        else:
            raise BaseException

    def is_leaf(self):
        return self.__right_child is None and \
               self.__left_child is None

    def level_of(self):
        if self.is_root():
            return 0
        else:
            return 1 + self.parent().level_of()

    def height(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + max(self.__left_child.height(),
                       self.__right_child.height())

    def inorder(self, order: list = None):
        order = [] if order is None else order
        if self.is_leaf():
            order.append(self.__element)
        else:
            if self.__left_child is not None:
                self.left_child().inorder(order)
            order.append(self.__element)
            if self.__right_child is not None:
                self.__right_child.inorder(order)
        return order

    def preorder(self, order: list = None):
        order = [] if order is None else order
        order.append(self.__element)
        for node in filter(lambda x: x is not None, (self.left_child(), self.__right_child)):
            node.preorder(order)
        return order

    def postorder(self, order: list = None):
        order = [] if order is None else order
        for node in filter(lambda x: x is not None, (self.left_child(), self.__right_child)):
            node.postorder(order)
        order.append(self.__element)
        return order
