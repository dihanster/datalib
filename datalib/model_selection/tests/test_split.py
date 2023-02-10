"""Test the split module"""
import numpy as np
import pytest
import warnings

from sklearn.utils._testing import ignore_warnings
from sklearn.utils.validation import _num_samples

from datalib.model_selection import BootstrapSplit


@ignore_warnings
def test_cross_validator_with_default_params():

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    X_1d = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 1, 2, 2, 2])
    groups = np.array([1, 2, 3, 4, 5])

    splitters = [BootstrapSplit(random_state=42)]
    reprs = ["BootstrapSplit(n_samples=5, n_splits=5, random_state=42)"]
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


def check_valid_split(train, test, n_samples=None):
    # Use python sets to get more informative assertion failure messages
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert train.intersection(test) == set()

    if n_samples is not None:
        # Check that the union of train an test split cover all the indices
        assert train.union(test) == set(range(n_samples))

def check_cv_coverage(cv, X, y, groups, expected_n_splits):
    n_samples = _num_samples(X)
    # Check that a all the samples appear at least once in a test fold
    assert cv.get_n_splits(X, y, groups) == expected_n_splits

    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)

def test_split_valueerrors():
    # Error when number of folds is <= 1
    with pytest.raises(ValueError):
        BootstrapSplit(n_splits = 0)

    # When n_splits is not integer:
    with pytest.raises(ValueError):
        BootstrapSplit(n_splits = 1.5)

def test_split_indices():
    # Check all indices are returned in the test folds
    X1 = np.ones(18)
    boot = BootstrapSplit(n_splits = 3)
    check_cv_coverage(boot, X1, y=None, groups=None, expected_n_splits=3)

    # Check all indices are returned in the test folds even when equal-sized
    # folds are not possible
    X2 = np.ones(17)
    boot = BootstrapSplit(3)
    check_cv_coverage(boot, X2, y=None, groups=None, expected_n_splits=3)

    # Check if get_n_splits returns the number of folds
    assert 5 == BootstrapSplit(5).get_n_splits(X2)

