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
