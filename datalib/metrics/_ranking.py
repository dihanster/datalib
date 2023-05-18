"""
Module containing the main metrics for ranking.
"""
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics._base import (
    _average_binary_score,
    _average_multiclass_ovo_score,
    _check_pos_label_consistency,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils._encode import _encode, _unique
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight


def cap_curve(
    y_true,
    y_score,
    sample_weight=None,
):
    """
    Base method to calculate the Cumulative Accuracy Profile Curve (CAP Curve).
    This metric ponders the rate of positive samples and the percentage of the
    dataset covered by each sequential cut-off threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a model / decision function.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    Returns
    -------
    cumulative_gains : ndarray of shape (n_samples,)
        Cumulative gain with each threshold (percentage of class 1).

    thresholds : ndarray of shape (n_samples,)
        Increasing thresholds (percentage of examples) on the decision
        function used to compute cap curve.

    gini: ndarray of shape (n_samples,)
        The normalized gini coefficient, calculated from the AUC.
    """
    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    sample_weight = _check_sample_weight(
        sample_weight, y_true, only_non_negative=True
    )

    if y_type == "binary":
        ranking = np.argsort(y_score)[::-1]
        ranked = y_true[ranking] * sample_weight

        cumulative_gains = np.append(0, np.cumsum(ranked) / np.sum(ranked))
        thresholds = np.arange(0, len(ranked) + 1) / len(ranked)

        # TODO: Implement traditional gini calculation
        gini = (
            2 * roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        ) - 1

        return cumulative_gains, thresholds, gini

    raise NotImplementedError("Only binary class supported!")


def delinquency_curve(y_true, y_score, pos_label=None):
    """Delinquency curve for a binary classification.

    The delinquency curve shows how the default rate (proportion of
    pos_labels) changes with different approval rates. The curve is
    typically plotted on a graph, with the approval rate on the x-axis
    and the default rate on the y-axis. The curve is created by sorting
    the samples by score and calculating the default rate for
    subsequently larger population getting the best scores at first.

    Deliquency curve is key on many actuarial operations, where grasping
    the relative percentage of misclassification on approval levels is
    vital.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1},
        then pos_label should be explicitly given.

    y_score : array-like, shape (n_samples,)
        Target scores, can either be probability estimates of the
        positive class, confidence values, or non-thresholded measure of
        decisions (as returned by "decision_function" on some
        classifiers).

    pos_label : int or str, default=None
        The label of the positive class.

    Returns
    -------
    approval_rate: array-like, shape (n_samples,).
        Increasing approval rate (percentage of approved best scores)
        used to compute `default_rate`. It lies in the support (0, 1).

    default_rate: array-like, shape (n_samples,).
        Default rates values for the approval rates such that the
        element i it proportion of delinquents when approving
        `approval_rate[i]` of the population.

    optimal_rate: array-like, shape (n_samples,).
        Optimal default rates for a perfect model.
    """
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided {labels}."
        )
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    check_consistent_length(y_true, y_score)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    y_true = y_true == pos_label

    scores_idxs = np.argsort(y_score)[::1]
    actual_idxs = np.argsort(y_true)[::1]

    y_true_sorted_by_scores = y_true[scores_idxs].copy()
    y_true_sorted = y_true[actual_idxs].copy()

    list_index = np.arange(1, len(y_true_sorted_by_scores) + 1)
    approval_rate = np.append(0, list_index / len(list_index))
    default_rate = np.append(0, y_true_sorted_by_scores.cumsum() / list_index)
    optimal_rate = np.append(0, y_true_sorted.cumsum() / list_index)

    return approval_rate, default_rate, optimal_rate

  
def ks_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    multi_class="raise",
    labels=None,
):
    """Compute the Kolmogorov-Smirnov statistic using the Youden's J statistic
    of the Receiver Operating Characteristic Curve (ROC AUC) from prediction
    scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see Parameters).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

        * In the binary case, it corresponds to an array of shape
          `(n_samples,)`. Both probability estimates and non-thresholded
          decision values can be provided. The probability estimates correspond
          to the **probability of the class with the greater label**,
          i.e. `estimator.classes_[1]` and thus
          `estimator.predict_proba(X, y)[:, 1]`. The decision values
          corresponds to the output of `estimator.decision_function(X, y)`.
          See more information in the :ref:`User guide <roc_auc_binary>`;
        * In the multiclass case, it corresponds to an array of shape
          `(n_samples, n_classes)` of probability estimates provided by the
          `predict_proba` method. The probability estimates **must**
          sum to 1 across the possible classes. In addition, the order of the
          class scores must correspond to the order of ``labels``,
          if provided, or else to the numerical or lexicographical order of
          the labels in ``y_true``;
        * In the multilabel case, it corresponds to an array of shape
          `(n_samples, n_classes)`. Probability estimates are provided by the
          `predict_proba` method and the non-thresholded decision values by
          the `decision_function` method. The probability estimates correspond
          to the **probability of the class with the greater label for each
          output** of the classifier.

    average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages. For multiclass targets, `average=None` is only
        implemented for `multi_class='ovo'` and `average='micro'` is only
        implemented for `multi_class='ovr'`.

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
        Only used for multiclass targets. Determines the type of configuration
        to use. The default value raises an error, so either
        ``'ovr'`` or ``'ovo'`` must be passed explicitly.

        ``'ovr'``:
            Stands for One-vs-rest. Computes the AUC of each class
            against the rest [3]_ [4]_. This
            treats the multiclass case in the same way as the multilabel case.
            Sensitive to class imbalance even when ``average == 'macro'``,
            because class imbalance affects the composition of each of the
            'rest' groupings.
        ``'ovo'``:
            Stands for One-vs-one. Computes the average AUC of all
            possible pairwise combinations of classes [5]_.
            Insensitive to class imbalance when
            ``average == 'macro'``.

    labels : array-like of shape (n_classes,), default=None
        Only used for multiclass targets. List of labels that index the
        classes in ``y_score``. If ``None``, the numerical or lexicographical
        order of the labels in ``y_true`` is used.

    Returns
    -------
    ks : float
        KS score.

    References
    ----------
    .. [1] `On the equivalence between Kolmogorov-Smirnov and ROC curve metrics
            for binary classification. Adeodato, P., Melo, S. (2016)
            <https://arxiv.org/pdf/1606.00496.pdf>`_

    .. [2] `Wikipedia entry for the Kolmogorov–Smirnov test
            <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>`_

    .. [3] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    Examples
    --------
    Binary case:

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from datalib.metrics import ks_score
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> ks_score(y, clf.predict_proba(X)[:, 1])
    0.92...
    >>> ks_score(y, clf.decision_function(X))
    0.92...

    Multiclass case:

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear").fit(X, y)
    >>> ks_score(y, clf.predict_proba(X), multi_class='ovr')
    0.95...

    Multilabel case:

    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> X, y = make_multilabel_classification(random_state=0)
    >>> clf = MultiOutputClassifier(clf).fit(X, y)
    >>> # get a list of n_output containing probability arrays of shape
    >>> # (n_samples, n_classes)
    >>> y_pred = clf.predict_proba(X)
    >>> # extract the positive columns for each output
    >>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    >>> ks_score(y, y_pred, average=None)
    array([0.52..., 0.62..., 0.80..., 0.57..., 0.79...])
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> clf = RidgeClassifierCV().fit(X, y)
    >>> ks_score(y, clf.decision_function(X), average=None)
    array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
    """

    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type == "multiclass" or (
        y_type == "binary" and y_score.ndim == 2 and y_score.shape[1] > 2
    ):
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return _multiclass_ks_score(
            y_true, y_score, labels, multi_class, average, sample_weight
        )
    elif y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        return _average_binary_score(
            _binary_ks_score,
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )
    else:  # multilabel-indicator
        return _average_binary_score(
            _binary_ks_score,
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )


def _binary_ks_score(y_true, y_score, sample_weight=None):
    """Binary KS score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.

    y_score : array-like of shape (n_samples,)
        Target scores.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. KS score "
            "is not defined in that case."
        )

    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    return np.max(tpr - fpr)


def _multiclass_ks_score(
    y_true, y_score, labels, multi_class, average, sample_weight
):
    """Multiclass KS score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

    labels : array-like of shape (n_classes,) or None
        List of labels to index ``y_score`` used for multiclass. If ``None``,
        the lexical order of ``y_true`` is used to index ``y_score``.

    multi_class : {'ovr', 'ovo'}
        Determines the type of multiclass configuration to use.

        ``'ovr'``:
            Calculate metrics for the multiclass case using the one-vs-rest
            approach.
        ``'ovo'``:
            Calculate metrics for the multiclass case using the one-vs-one
            approach.

    average : {'micro', 'macro', 'weighted'}
        Determines the type of averaging performed on the pairwise binary
        metric scores.

        ``'micro'``:
            Calculate metrics for the binarized-raveled classes. Only supported
            for `multi_class='ovr'`.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.
    """
    # validation of the input y_score
    if not np.allclose(1, y_score.sum(axis=1)):
        raise ValueError(
            "Target scores need to be probabilities for multiclass "
            "KS, i.e. they should sum up to 1.0 over classes"
        )

    # validation for multiclass parameter specifications
    average_options = ("macro", "weighted", None)
    if multi_class == "ovr":
        average_options = ("micro",) + average_options
    if average not in average_options:
        raise ValueError(
            "average must be one of {0} for multiclass problems".format(
                average_options
            )
        )

    multiclass_options = ("ovo", "ovr")
    if multi_class not in multiclass_options:
        raise ValueError(
            "multi_class='{0}' is not supported "
            "for multiclass KS, multi_class must be "
            "in {1}".format(multi_class, multiclass_options)
        )

    if average is None and multi_class == "ovo":
        raise NotImplementedError(
            "average=None is not implemented for multi_class='ovo'."
        )

    if labels is not None:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of given labels, {0}, not equal to the number "
                "of columns in 'y_score', {1}".format(
                    len(classes), y_score.shape[1]
                )
            )
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError(
                "'y_true' contains labels not in parameter 'labels'"
            )
    else:
        classes = _unique(y_true)
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of classes in y_true not equal to the number of "
                "columns in 'y_score'"
            )

    if multi_class == "ovo":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is not supported "
                "for multiclass one-vs-one KS, "
                "'sample_weight' must be None in this case."
            )
        y_true_encoded = _encode(y_true, uniques=classes)
        # Hand & Till (2001) implementation (ovo)
        return _average_multiclass_ovo_score(
            _binary_ks_score, y_true_encoded, y_score, average=average
        )
    else:
        # ovr is same as multi-label
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return _average_binary_score(
            _binary_ks_score,
            y_true_multilabel,
            y_score,
            average,
            sample_weight=sample_weight,
        )


def numpy_fill(arr):
    """Solution provided by Divakar.
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out


def scipy_inspired(data1, data2, wei1, wei2):
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


def ks_curve(y_true, y_score, *, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true, input_name="y_true")
    if not (
        y_type == "binary"
        or (y_type == "multiclass" and pos_label is not None)
    ):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    mask_pos = y_true == pos_label
    z1 = y_score[mask_pos]
    z0 = y_score[~mask_pos]

    if sample_weight is not None:
        w1 = sample_weight[mask_pos]
        w0 = sample_weight[~mask_pos]
    else:
        w1 = np.ones(z1.shape)
        w0 = np.ones(z0.shape)

    acum1, acum0, thresholds = scipy_inspired(z1, z0, w1, w0)
    return acum1, acum0, thresholds
