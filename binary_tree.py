

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

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, parent):
        self.__parent = parent

    def is_root(self):
        return self.parent() is None

    @property
    def left_child(self):
        return self.__left_child

    @left_child.setter
    def left_child(self, left_child):
        self.__left_child = left_child

    @property
    def right_child(self):
        return self.__right_child

    @right_child.setter
    def right_child(self, right_child):
        self.__right_child = right_child

    def append_left_child(self, child):
        child.__parent = self
        if self.__left_child is None:
            self.__left_child = child
        else:
            raise IndexError("the tree has already a left child")

    def append_right_child(self, child):
        child.__parent = self
        if self.__right_child is None:
            self.__right_child = child
        else:
            raise IndexError("the tree has already a right child")

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
                self.left_child.inorder(order)
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

    @staticmethod
    def arithmetic_expr_to_btree(expr: str) -> 'BinaryTree':
        """
        Pablo's solution
        :param expr: must have all the parentheses even if there is no ambiguity
        :return: a binary tree of the expression
        """
        expr_tree = BinaryTree()
        operators = {"+", "-", "*", "/"}
        current_digit = ""
        current_node = expr_tree
        for c in expr:
            if c.isdigit() or c == ".":
                current_digit += c
            elif current_digit:
                if current_node.left_child is None:
                    current_node.append_left_child(BinaryTree(current_digit))
                else:
                    current_node.append_right_child(BinaryTree(current_digit))
                current_digit = ""

            if c in operators:
                current_node.element = c
            elif c == "(":
                if current_node.left_child is None:
                    current_node.append_left_child(BinaryTree())
                    current_node = current_node.left_child
                else:
                    current_node.append_right_child(BinaryTree())
                    current_node = current_node.right_child

            elif c == ")":
                current_node = current_node.parent

        return expr_tree

    @staticmethod
    def arithmetic_expression_to_binary_tree(expression):
        """teacher's solution"""
        exp_tree = BinaryTree()
        digit = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        operator = {'+', '-', '*', '/'}
        current_node = exp_tree

        for ele in expression:
            if ele == '(':
                aux_node = BinaryTree()
                aux_node.parent = current_node
                current_node.left_child = aux_node
                current_node = aux_node
            elif ele in digit:
                current_node.element = ele
                current_node = current_node.parent
            elif ele in operator:
                current_node.element = ele
                aux_node = BinaryTree()
                current_node.right_child = aux_node
                aux_node.parent = current_node
                current_node = aux_node
            else:
                current_node = current_node.parent
        return exp_tree

    def __str__(self):
        return str(self.inorder())


if __name__ == "__main__":
    print(BinaryTree.arithmetic_expr_to_btree("(19.7+2.6)*(35+4)").inorder())
    print(BinaryTree.arithmetic_expression_to_binary_tree("((1+2)*(3+4))").inorder())
