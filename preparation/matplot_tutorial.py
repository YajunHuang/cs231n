"""
Matplotlib is a plotting library.   http://matplotlib.org/
In this section give a brief introduction to the matplotlib.pyplot module,
which provides a plotting system similar to that of MATLAB.
"""
import numpy as np
from matplotlib import pyplot as plt


def exercise_1():
    """
    Plotting
    :return:
    """
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()


def exercise_2():
    """
    Subplot
    :return:
    """

if __name__ == '__main__':
    exercise_1()