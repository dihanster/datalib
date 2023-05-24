import numpy as np

from numpy.testing import assert_array_equal
from datalib.utils import _transform_label_vector_to_matrix


def test__transformer():
    multi_y_true = np.array([2, 1, 0])
    matrix = _transform_label_vector_to_matrix(multi_y_true)
    assert_array_equal(matrix, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
