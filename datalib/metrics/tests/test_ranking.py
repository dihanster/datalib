import numpy as np
import pytest

from numpy.testing import assert_array_equal

from datalib.metrics import cap_curve


def test_cap_curve__success_case():

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    cumulative_gain, thresholds, gini = cap_curve(y_true, y_scores)

    assert_array_equal(cumulative_gain, np.array([0, 0.5, 0.5, 1.0, 1.0]))
    assert_array_equal(thresholds, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert gini == 0.5


def test_cap_curve__multiclass_exception():

    y_true = np.array([0, 0, 1, 1, 2])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8, 0.04])

    with pytest.raises(Exception) as exc_info:
        _, _, _ = cap_curve(y_true, y_scores)

    assert (
        str(exc_info.value)
        == "Only binary class supported!"
    )
