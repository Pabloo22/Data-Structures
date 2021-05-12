

class Tree:
    __parent: 'Tree' or None
    __element: any
    __children: list['Tree']

    def __init__(self, ele=None):
        self.__parent = None
        self.__element = ele
        self.__children = []

    @property
    def element(self):
        return self.__element

    @element.setter
    def element(self, ele):
        self.__element = ele

    @property
    def parent(self):
        return self.__parent

    @property
    def children(self):
        return self.__children

    def is_root(self):
        return self.parent() is None

    def append_child(self, child: 'Tree'):
        self.__children.append(child)

    def is_leave(self):
        return not self.__children

    def level_of(self):
        if self.is_root():
            return 0
        else:
            return 1 + self.parent().level_of()

    def height(self):
        if self.is_leave():
            return 1
        else:
            return 1 + max(map(lambda x: x.height(), self.__children))
