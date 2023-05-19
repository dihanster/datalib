import numpy as np


def _transform_label_vector_to_matrix(data_vector):
    """
    Base method to transform a vector with `m` different ordinal classes
    into a binary matrix with shape (n_samples, m), on a one-hot encoding
    fashion.

    Parameters
    ----------
    data_vector : ndarray of shape (n_samples,)
        A vector with values ordinal values.

    """
    base_array = np.max(data_vector) + 1
    base_array = np.eye(base_array)[data_vector]

    return base_array
