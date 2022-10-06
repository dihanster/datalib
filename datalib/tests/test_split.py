"""Test the split module"""
import pytest
import warnings
import numpy as np

from sklearn.utils._testing import ignore_warnings
from ..model_selection.model_selection import BootstrapSplit


@ignore_warnings
def test_cross_validator_with_default_params():
    n_samples = 4
    n_unique_groups = 4
    n_splits = 2
    p = 2
    n_shuffle_splits = 10  # (the default value)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    X_1d = np.array([1, 2, 3, 4])
    y = np.array([1, 1, 2, 2])
    groups = np.array([1, 2, 3, 4])

    splitters = [
        BootstrapSplit(random_state = 0)
    ]
    reprs = [
        "BootstrapSplit(n_splits=5, random_state=0)"
    ]
    n_splits_expected = [
        5
    ]
    for i, (splitter, splitter_repr) in enumerate(zip(splitters, reprs)):
        # Test if get_n_splits works correctly
        assert n_splits_expected[i] == splitter.get_n_splits(X, y, groups)

        # Test if the cross-validator works as expected even if
                # the data is 1d
        np.testing.assert_equal(
            list(splitter.split(X, y, groups)), list(splitter.split(X_1d, y, groups))
        )

        # Test if the repr works without any errors
        assert splitter_repr == repr(splitter)

        # Test that train, test indices returned are integers
        for train, test in splitter.split(X, y, groups):
            assert np.asarray(train).dtype.kind == "i"
            assert np.asarray(test).dtype.kind == "i"


def test_2d_y():
    # smoke test for 2d y and multi-label
    n_samples = 30
    rng = np.random.RandomState(1)
    X = rng.randint(0, 3, size=(n_samples, 2))
    y = rng.randint(0, 3, size=(n_samples,))
    y_2d = y.reshape(-1, 1)
    y_multilabel = rng.randint(0, 2, size=(n_samples, 3))
    groups = rng.randint(0, 3, size=(n_samples,))
    splitters = [
        BootstrapSplit()
    ]
    for splitter in splitters:
        list(splitter.split(X, y, groups))
        list(splitter.split(X, y_2d, groups))
        try:
            list(splitter.split(X, y_multilabel, groups))
        except ValueError as e:
            allowed_target_types = ("binary", "multiclass")
            msg = "Supported target types are: {}. Got 'multilabel".format(
                allowed_target_types
            )
            assert msg in str(e)