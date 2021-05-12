from binary_tree import BinaryTree


class BSTree(BinaryTree):

    def __init__(self, key: int or float = None, element: any = None):
        super().__init__((key, element))

    def __is_higher(self, key: int or float) -> bool:
        return self.element[0] > key

    def min(self) -> tuple[int or float, any]:
        if self.left_child is None:
            return self.element
        return self.left_child.min()

    def max(self) -> tuple[int or float, any]:
        if self.right_child is None:
            return self.element

        return self.right_child.max()

    def find(self, key: int or float) -> 'BSTree' or None:
        if self.element[0] == key:
            return self

        if self.element[0] > key and self.left_child is not None:
            return self.left_child.find(key)
        elif self.right_child is not None:
            return self.right_child.find(key)
        else:
            return None

    def get_value(self, key: int or float) -> any:
        node = self.find(key)
        return node.element[1] if node is not None else None

    def insert(self, key: int or float, value: any):
        if self.__is_higher(key):
            if self.left_child is None:
                new_node = BSTree(key, value)
                self.append_left_child(new_node)
            else:
                self.left_child.insert(key, value)
        else:
            if self.right_child is None:
                new_node = BSTree(key, value)
                self.append_right_child(new_node)
            else:
                self.right_child.insert(key, value)

    def remove(self, key: int or float):
        aux_node = self.find(key)

        if aux_node.is_leave():
            if aux_node.parent.left_child == aux_node:
                aux_node.parent.left_child = None
            else:
                aux_node.parent.right_child = None

        elif aux_node.left_child is None:  # right node is not empty
            if aux_node.parent.left_child == aux_node:
                aux_node.parent.left_child = aux_node.right_child
            else:
                aux_node.parent.right_child = aux_node.right_child
            aux_node.right_child = None

        elif aux_node.right_child is None:  # left node is not empty
            if aux_node.parent.left_child == aux_node:
                aux_node.parent.left_child = aux_node.left_child
            else:
                aux_node.parent.right_child = aux_node.left_child
            aux_node.left_child = None

        else:  # both children are not empty
            predecessor_key, _ = aux_node.left_child.max()
            predecessor_element = aux_node.left_child.find(predecessor_key).element
            aux_node.element = predecessor_element
            aux_node.left_child.remove(predecessor_key)
