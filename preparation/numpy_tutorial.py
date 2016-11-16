"""
 A numpy tutorial following http://cs231n.github.io/python-numpy-tutorial/#numpy
"""


import numpy as np


def exercise_1():
    a = np.array([1, 2, 3])
    print type(a)
    print a.shape
    print a[0], a[1], a[2]
    a[0] = 5
    print a

    b = np.array([[1,2,3],[4,5,6]])
    print b.shape
    print b[0][0], b[0][1], b[1][0]
    print b[0, 0], b[0, 1], b[1, 0]


def exercise_2():
    a = np.zeros((2,2))
    print a

    b = np.ones((1,2))
    print b

    c = np.full((2,2), 7, dtype='int32')
    print c

    d = np.eye(2)
    print d

    e = np.random.random((2,2))
    print e


def exercise_3():
    """
    Array indexing: Slicing, Integer array indexing
    :return:
    """
    a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    b = a[:2, 1:3]

    print a[0, 1]
    b[0, 0] = 77
    print a[0, 1]

    row_r1 = a[1, :]
    row_r2 = a[1:2, :]
    print row_r1, row_r1.shape  # Prints "[5 6 7 8] (4,)"
    print row_r2, row_r2.shape  # Prints "[[5 6 7 8]] (1, 4)"

    col_r1 = a[:, 1]
    col_r2 = a[:, 1:2]
    print col_r1, col_r1.shape  # Prints "[ 2  6 10] (3,)"
    print col_r2, col_r2.shape  # Prints "[[ 2]
                                #          [ 6]
                                #          [10]] (3, 1)"

    a = np.array([[1,2], [3, 4], [5, 6]])
    print a[[0,1,2], [0,1,0]]
    print np.array([a[0, 0], a[1, 1], a[2, 0]])
    print a[[0,0], [1,1]]
    print np.array([a[0,1], a[0,1]])

    a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
    b = np.array([0, 2, 0, 1])
    print a[np.arange(4), b]

    a[np.arange(4), b] += 10
    print a


def exercise_4():
    """
    Array indexing: Boolean array indexing
    :return:
    """
    a = np.array([[1,2], [3, 4], [5, 6]])
    bool_idx = (a > 2)
    print bool_idx

    print a[bool_idx]
    print a[a > 2]


def exercise_5():
    """
    Data types
    :return:
    """
    x = np.array([1, 2])
    print x.dtype

    x = np.array([1.0, 2.0])
    print x.dtype

    x = np.array([1, 2], dtype=np.int64)
    print x.dtype


def exercise_6():
    """
    Array math: elementwise operations
    :return:
    """
    x = np.array([[1,2],[3,4]], dtype=np.float64)
    y = np.array([[5,6],[7,8]], dtype=np.float64)

    print x + y
    print np.add(x, y)

    print x - y
    print np.subtract(x, y)

    print x * y
    print np.multiply(x, y)

    print x / y
    print np.divide(x, y)

    print np.sqrt(x)


def exercise_7():
    """
    inner product
    :return:
    """
    x = np.array([[1,2],[3,4]])
    y = np.array([[5,6],[7,8]])

    v = np.array([9,10])
    w = np.array([11, 12])

    print v.dot(w)
    print np.dot(v, w)

    # Matrix / vector product; both produce the rank 1 array [29 67]
    print x.dot(v)
    print np.dot(x, v)

    # Matrix / matrix product; both produce the rank 2 array
    # [[19 22]
    #  [43 50]]
    print x.dot(y)
    print np.dot(x, y)


def exercise_8():
    """
    Operations on array
    :return:
    """
    x = np.array([[1,2],[3,4]])
    print np.sum(x)
    print np.sum(x, axis=0)
    print np.sum(x, axis=1)

    print x.T

    v = np.array([1, 2, 3, 4])
    print v
    print v.T


def exercise_9():
    """
    Broadcasting.
    :return:
    """
    x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
    v = np.array([1, 0, 1])
    y = np.empty_like(x)

    for i in range(4):
        y[i, :] = x[i, :] + v
    print y

    vv = np.tile(v, (4, 1))
    y = x + vv
    print y

    y = x + v
    print y


def exercise_10():
    """
    Broadcasting. Functions that support broadcasting are known as universal functions.
    You can find the list of all universal functions in the documentation.
    http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs
    :return:
    """
    # Compute outer product of vectors
    v = np.array([1,2,3])  # v has shape (3,)
    w = np.array([4,5])    # w has shape (2,)
    print np.reshape(v, (3, 1)) * w

    # Add a vector to each row of a matrix
    x = np.array([[1,2,3], [4,5,6]])
    print x + v

    # Add a vector to each column of a matrix
    print (x.T + w).T
    print x + np.reshape(w, (2, 1))

    # Multiply a matrix by a constant:
    print x * 2


if __name__ == '__main__':
    exercise_9()
