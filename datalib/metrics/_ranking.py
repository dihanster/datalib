<<<<<<< HEAD
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target

import numpy as np


def transform_label_vector_to_matrix(y_true):
    base_array = np.max(y_true) + 1
    base_array = np.eye(base_array)[y_true]

    return base_array


def discrete_ranked_probability_loss(y_true, y_proba):
    """Discrete RPS Score.

    The Discrete RPS Loss function leverages the RPS as a loss function. This loss function wraps up
    the unitary RPS calculation, reduce it to all realizations of an response vector y_true and
    its predicted probabilities y_proba, and outputs a mean across all unitary RPS.
    """

    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_proba = check_array(y_proba, ensure_2d=False)
    check_consistent_length(y_true, y_proba)

    y_true_matrix = transform_label_vector_to_matrix(y_true)

    bias_correction = len(set(y_true)) - 1

    if y_proba.shape[1] != len(set(y_true)):
        raise ValueError(
            f"Number of unique labels and columns on y_proba don't match. \
             Provided labels {set(y_true)}."
        )

    return np.average(
        [
            discrete_ranked_probability_score(y_score, y_true, bias_correction)
            for y_score, y_true in zip(y_proba, y_true_matrix)
        ]
    )


def discrete_ranked_probability_score(y_true, y_scores, bias_correction=1):
    """Ranked Probability Score.

    The Unbiased RPS is used to quantify the performance of probabilistic prediction systems.
    It compares the cumulative density function of a probabilistic forecast with a ground truth,
    condined by a number of levels(Categories). Brier Score is the two-level case of RPS.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Correct labels for given dataset.
    y_proba : array of shape (n_samples, n_classes)
        Probability estimates provided by `predict_proba` method.
    Returns
    -------
    approval_rate: array.
        An array containing the approval rates used to compute the default_rate curve.
    default_rate: array.
        An array containing the default rates values for the approval rates provided in approval_rate.
    optimal_rate: array.
        An array containing the optimal default rates for a perfect model.
    References
    ----------
    .. [1] `The Discrete Brier and Ranked Probability Skill Scores
            <https://journals.ametsoc.org/view/journals/mwre/135/1/mwr3280.1.xml>`_
    .. [2] ` Forecast Verification - Issues, Methods and FAQ
            <https://www.cawcr.gov.au/projects/verification/verif_web_page.html#RPS>`_
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from datalib.metrics import discrete_ranked_probability_score
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> clf = LogisticRegression(random_state=0)
    >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
    >>> y_proba = clf.predict_proba(X_test)
    >>> discrete_ranked_probability_score(y_test, y_proba)
    """
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_proba = check_array(y_proba, ensure_2d=False)  
    check_consistent_length(y_true, y_proba)    


    if y_proba.shape[1] != len(set(y_true)):
        raise ValueError(
            f"Number of unique labels and columns on y_proba don't match. \
             Provided labels {set(y_true)}."
        )
    """
    if y_scores.shape != y_true.shape:
        raise ValueError(f"Matrices y_true and y_proba format don't match.")

    y_true_cumsum = np.cumsum(y_true, axis=0)
    y_scores_cumsum = np.cumsum(y_scores, axis=0)

    return np.sum((y_scores_cumsum - y_true_cumsum) ** 2) / bias_correction
=======
"""
Module containing the main metrics for ranking.
"""
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.utils import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
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
>>>>>>> main
