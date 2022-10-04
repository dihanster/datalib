import numbers
import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.model_selection._split import _build_repr
from sklearn.utils import check_random_state, indexable, resample
from sklearn.utils.validation import _num_samples

__all__ = [
    "BaseBootstrapSplit",
    "BootstrapSplit"
]


class BaseBootstrapSplit(metaclass = ABCMeta):
    """Base class for BootstrapSplit and StratifiedBootstrapSplit
    
    Implementations must define `_iter_indices`
    """
    
    def __init__(self, n_splits: int = 10, *,  random_state: int = None):
        self.n_splits = n_splits
        self.random_state = random_state
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "Bootstrap Split requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )
    
    def split(self, X, y = None, groups = None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test
        
    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def __repr__(self):
        return _build_repr(self)


class BootstrapSplit(BaseBootstrapSplit):
    """Bootstrap K-Folds cross-validator

    Provides train/test indices to split data in bootstraped train/test sets.
    The folds are determined by the number of bootstrap iterations. 
    
    At each bootstrap round, the train folds are nothing else than the boostrapped samples
    of the dataset whereas the test sets are composed of all observations that 
    are missing from the train folds.
    
    Parameters
    ----------
    n_splits: int, default=10
        Number of bootstrap rounds. Must at least be 2.
    shuffle: bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
    
    Examples
    --------
    >>> import numpy as np
    >>> from datalib.model_selection import BootstrapSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> boot = BootstrapSplit()
    >>> boot.get_n_splits(X)
    5
    >>> print(boot)
    BootstrapSplit(n_splits=5, random_state=None)
    >>> for train_index, test_index in boot.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 3 4 1 1 4] TEST: [2, 5]
    TRAIN: [0 2 0 3 2 0] TEST: [1, 4, 5]
    TRAIN: [4 1 4 0 2 4] TEST: [3, 5]
    TRAIN: [5 5 0 3 3 2] TEST: [1, 4]
    TRAIN: [2 2 4 0 3 2] TEST: [1, 5]
    
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """
    def __init__(self, n_splits: int = 5, *, random_state=None):
        super().__init__(n_splits=n_splits, random_state=random_state)

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        # maybe we need a validate_bootstrap_split function here TBD

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # generate a bootstrap sample
            train_index = resample(indices, replace = True, random_state = rng)
            test_index = list(set(indices).difference(set(train_index)))
            # assert the test_index is not empty
            assert len(test_index) != 0, f'Test set is empty for bootstrap round {i}'
            yield train_index, test_index

