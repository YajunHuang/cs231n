"""
Numpy provides a high-performance multidimensional array and basic tools to compute with and manipulate these arrays.
SciPy builds on this, and provides a large number of functions that operate on numpy arrays and are useful for different
types of scientific and engineering applications.
The best way to get familiar with SciPy is to http://docs.scipy.org/doc/scipy/reference/index.html
"""

from scipy.misc import imread, imsave, imresize

def exercise_1():
    """
    Image operations
    :return:
    """
    image = imread('./cat.jpg')
    print image.dtype, image.shape

    image_tinted = image * [1, 0.95, 0.9]
    image_tinted = imresize(image_tinted, (300, 300))

    imsave('./cat_tinted.jpg', image_tinted)


def exercise_2():
    """
    Image operations: http://docs.scipy.org/doc/scipy/reference/io.html
    :return:
    """
    pass


if __name__ == '__main__':
    exercise_1()