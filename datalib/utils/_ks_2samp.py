import numpy as np


def numpy_fill(arr):
    """
    TODO: docstring.
    Solution provided by Divakar.
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array # noqa
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out


def scipy_inspired_ks_2samp(data1, data2, wei1, wei2):
    """TODO: docstring."""
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)

    min_data = np.min([data1[ix1[0]], data2[ix2[0]]])
    max_data = np.max([data1[ix1[-1]], data2[ix2[-1]]])

    data1 = np.hstack([min_data, data1[ix1], max_data])
    data2 = np.hstack([min_data, data2[ix2], max_data])
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.sort(np.concatenate([data1, data2]))
    cwei1 = np.hstack([min_data, np.cumsum(wei1) / np.sum(wei1), max_data])
    cwei2 = np.hstack([min_data, np.cumsum(wei2) / np.sum(wei2), max_data])

    data = np.sort(np.concatenate([data1, data2]))
    distinct_value_indices = np.where(np.diff(data))[0]
    threshold_idxs = np.r_[distinct_value_indices, data.size - 1]

    dic1 = dict(zip(data1, cwei1))
    dic1.update({min_data: 0, max_data: 1})
    y1 = np.array(list(map(dic1.get, data[threshold_idxs])))
    y1 = numpy_fill(y1.astype(float))

    dic2 = dict(zip(data2, cwei2))
    dic2.update({min_data: 0, max_data: 1})
    y2 = np.array(list(map(dic2.get, data[threshold_idxs])))
    y2 = numpy_fill(y2.astype(float))

    return y1, y2, data[threshold_idxs]
