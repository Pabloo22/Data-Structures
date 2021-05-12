from copy import deepcopy


class Matrix2D:

    __array: list[list[int or float]]
    __dim: tuple[int, int]

    def __init__(self, array: list[list[int or float]]):
        for row in array:
            if len(row) != len(array[0]):
                raise ValueError("creating a matrix from ragged nested sequences")

        self.__array = array
        self.__dim = (len(array), len(array[0]))

    @property
    def dim(self):
        return self.__dim

    def __iter__(self):
        return iter(self.__array)

    def __str__(self):
        s = ""
        for row in self:
            s += str(row) + "\n"
        return s

    def __getitem__(self, index) -> list[int or float]:
        return self.__array[index]

    def __setitem__(self, index, value):
        self.__array[index] = value

    def __add__(self, other: 'Matrix2D'):
        array = [[self.__array[i][j] + other[i][j] for j in range(self.__dim[1])] for i in range(self.__dim[0])]
        return Matrix2D(array)

    def __sub__(self, other):
        array = [[self.__array[i][j] - other[i][j] for j in range(self.__dim[1])] for i in range(self.__dim[0])]
        return Matrix2D(array)

    def __mul__(self, other):
        if isinstance(other, Matrix2D):
            array = [[self.__array[i][j] * other[i][j] for j in range(self.__dim[1])] for i in range(self.__dim[0])]
        else:
            array = [[self.__array[i][j] * other for j in range(self.__dim[1])] for i in range(self.__dim[0])]

        return Matrix2D(array)

    def __matmul__(self, other: 'Matrix2D'):
        if self.dim[1] != other.dim[0]:
            raise ValueError("the number of columns in the first matrix must be equal to the number of rows "
                             "in the second matrix")

        return Matrix2D.strassen_algorithm(self, other)

    def __pow__(self, power, modulo=None):

        actual = self
        for _ in range(power - 1):
            actual @= self

        return actual

    def copy(self) -> 'Matrix2D':
        return deepcopy(self)

    def to_list(self) -> list[list]:
        return deepcopy(self.__array)

    @staticmethod
    def resize(matrix: 'Matrix2D', dims: tuple[int, int] or list[int, int]) -> 'Matrix2D':
        """
        adds zeros to expand the matrix or eliminates rows and columns until it fits with the given dimensions
        :return: a new Matrix
        """
        new = matrix.to_list()

        # removing rows
        if dims[0] < matrix.dim[0]:
            for _ in range(matrix.dim[0] - dims[0]):
                new.pop()

        # adding rows
        elif dims[0] > matrix.dim[0]:
            for _ in range(dims[0] - matrix.dim[0]):
                new.append([0] * dims[1])

        # removing columns
        if dims[1] < matrix.dim[1]:
            for i in range(len(new)):
                for _ in range(matrix.dim[1] - dims[1]):
                    new[i].pop()

        # adding columns
        elif dims[1] > matrix.dim[1]:
            for i in range(matrix.dim[0]):
                new[i].extend([0] * (dims[1] - matrix.dim[1]))

        return Matrix2D(new)

    @staticmethod
    def square_even_resize(matrix: 'Matrix2D') -> 'Matrix2D':
        if matrix.dim[0] != matrix.dim[1] or matrix.dim[0] % 2 == 0:
            if matrix.dim[0] < matrix.dim[1]:
                if matrix.dim[1] % 2 == 0:
                    return Matrix2D.resize(matrix, (matrix.dim[1], matrix.dim[1]))
                else:
                    return Matrix2D.resize(matrix, (matrix.dim[1] + 1, matrix.dim[1] + 1))
            else:
                if matrix.dim[0] % 2 == 0:
                    return Matrix2D.resize(matrix, (matrix.dim[0], matrix.dim[0]))
                else:
                    return Matrix2D.resize(matrix, (matrix.dim[0] + 1, matrix.dim[0] + 1))

        return matrix

    @staticmethod
    def divide_into_four(matrix: 'Matrix2D') -> tuple['Matrix2D', 'Matrix2D',
                                                      'Matrix2D', 'Matrix2D']:

        if matrix.dim[0] != matrix.dim[1] or matrix.dim[0] % 2 == 1:
            raise ValueError("the matrix must be N x N with N an even number")

        m1 = Matrix2D([row[:len(row) // 2] for row in matrix[:matrix.dim[0] // 2]])
        m2 = Matrix2D([row[len(row) // 2:] for row in matrix[:matrix.dim[0] // 2]])
        m3 = Matrix2D([row[:len(row) // 2] for row in matrix[matrix.dim[0] // 2:]])
        m4 = Matrix2D([row[len(row) // 2:] for row in matrix[matrix.dim[0] // 2:]])

        return m1, m2, m3, m4

    @staticmethod
    def join(m1: 'Matrix2D', m2: 'Matrix2D', m3: 'Matrix2D', m4: 'Matrix2D') -> 'Matrix2D':
        
        array = m1.to_list()
        for row in m3:
            array.append(row)

        aux = m2.to_list()
        for row in m4:
            aux.append(row)

        for i, row in enumerate(array):
            row.extend(aux[i])

        return Matrix2D(array)

    @staticmethod
    def strassen_algorithm(matrix1: 'Matrix2D', matrix2: 'Matrix2D'):

        if matrix1.dim == matrix2.dim == (1, 1):
            return Matrix2D([[matrix1[0][0] * matrix2[0][0]]])

        out_dim = (matrix1.dim[0], matrix2.dim[1])
        matrix1 = Matrix2D.square_even_resize(matrix1)
        matrix2 = Matrix2D.square_even_resize(matrix2)

        a, b, c, d = Matrix2D.divide_into_four(matrix1)
        e, f, g, h = Matrix2D.divide_into_four(matrix2)

        p1 = Matrix2D.strassen_algorithm(a, f - h)
        p2 = Matrix2D.strassen_algorithm(a + b, h)
        p3 = Matrix2D.strassen_algorithm(c + d, e)
        p4 = Matrix2D.strassen_algorithm(d, g - e)
        p5 = Matrix2D.strassen_algorithm(a + d, e + h)
        p6 = Matrix2D.strassen_algorithm(b - d, g + h)
        p7 = Matrix2D.strassen_algorithm(a - c, e + f)

        c1 = p5 + p4 - p2 + p6
        c2 = p1 + p2
        c3 = p3 + p4
        c4 = p1 + p5 - p3 - p7

        sol = Matrix2D.join(c1, c2,
                            c3, c4)
        return Matrix2D.resize(sol, out_dim)


if __name__ == "__main__":

    import numpy as np

    a = Matrix2D([[0,   1,  2, 3],
                  [10, 11, 12, 13],
                  [20, 21, 22, 23],
                  [30, 31, 32, 33]])
    np_a = np.array(a.to_list())

    b = Matrix2D([[1, 2],
                  [2, 4]])
    np_b = np.array(b.to_list())

    c = Matrix2D([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    np_c = np.array(c.to_list())

    d = Matrix2D([[1],
                  [11],
                  [21],
                  [31]])
    np_d = np.array(d.to_list())

    assert (a ** 2).to_list() == (np_a @ np_a).tolist()
    assert (a @ d).to_list() == (np_a @ np_d).tolist()
