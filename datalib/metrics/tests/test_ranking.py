"""
Module with the test for the cap curve display functions.
"""
import numpy as np
import pytest

from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_almost_equal,
)
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_random_state

from datalib.metrics import (
    cap_curve,
    ranked_probability_score_loss,
    delinquency_curve,
)


@pytest.fixture(scope="module")
def iris_data():
    return datasets.load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    X, y = iris_data
    return X[y < 2], y[y < 2]


def make_prediction(dataset=None, binary=True, score=True):
    """
    Make some classification predictions on a toy dataset using a SVC model.
    """

    if dataset is None:
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary is True:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    clf.fit(X[:half], y[:half])

    if score is True:
        y_score = clf.predict_proba(X[half:])
        if binary is True:
            y_score = y_score[:, 1]
    else:
        y_score = clf.decision_function(X[half:])

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, y_score


def test_cap_curve():
    """
    Tests CAP Curve return attributes such as the Gini value and the
    shape of the cumulative values and thresholds, using scores from the
    `predict_proba` and `decision_function` of a model.
    """
    # With predict_proba
    y_true, _, y_score = make_prediction()
    expected_gini = (2 * roc_auc_score(y_true, y_score)) - 1

    cumulative_gain, thresholds, gini = cap_curve(y_true, y_score)

    assert_array_almost_equal(gini, expected_gini, decimal=2)
    assert cumulative_gain.shape == thresholds.shape

    # With decision function
    y_true, _, y_score = make_prediction(score=False)
    expected_gini = (2 * roc_auc_score(y_true, y_score)) - 1

    cumulative_gain, thresholds, gini = cap_curve(y_true, y_score)

    assert_array_almost_equal(gini, expected_gini, decimal=2)
    assert cumulative_gain.shape == thresholds.shape


def test_cap_curve_toy_data():
    """
    It tests a simple example for calculating the CAP Curve, comparing
    it with the expected value for the cumulative values, the threshols
    and the Gini value.
    """
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    cumulative_gain, thresholds, gini = cap_curve(y_true, y_scores)

    assert_array_equal(cumulative_gain, np.array([0, 0.5, 0.5, 1.0, 1.0]))
    assert_array_equal(thresholds, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert gini == 0.5


def test_cap_curve_multiclass_exception():
    """
    Tests a multiclass example case, which function should return an error
    for the CAP Curve.
    """
    y_true = np.array([0, 0, 1, 1, 2])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8, 0.04])

    with pytest.raises(Exception) as exc_info:
        _, _, _ = cap_curve(y_true, y_scores)

    assert str(exc_info.value) == "Only binary class supported!"


def test_gini():
    """
    Test Gini's calculation for a simple example.
    """
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])

    _, _, gini = cap_curve(y_true, y_scores)
    assert_array_almost_equal(gini, 0.5)


def test_cap_curve_gain_increasing():
    """
    Tests whether the cumulative gain attribute returned by
    the function of CAP Curve is increasing.
    """
    y_true, _, y_score = make_prediction()
    cumulative_gain, thresholds, _ = cap_curve(y_true, y_score)

    assert (np.diff(cumulative_gain) < 0).sum() == 0
    assert (np.diff(thresholds) < 0).sum() == 0


def test_cap_curve_end_points():
    """ "
    Tests whether the end and start points of the CAP curve's
    cumulative values are consistent with expectations.
    """
    y_true, _, y_score = make_prediction()
    cumulative_gain, _, _ = cap_curve(y_true, y_score)

    assert cumulative_gain[0] == 0
    assert cumulative_gain[-1] == 1


def test_cap_curve_sample_weight():
    """
    TODO: Implement a function to test the sample weight
    calculation for the CAP curve function.
    """


def test_delinquency_curve__initial_and_end_values():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    approval_rate, _, _ = delinquency_curve(y_true, y_scores)

    assert approval_rate[0] == 0
    assert approval_rate[-1] == 1


def test_delinquency_curve__success_case():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    approval_rate, default_rate, optimal_rate = delinquency_curve(
        y_true, y_scores
    )

    assert_array_equal(approval_rate, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert_almost_equal(
        default_rate, np.array([0, 0, 0.5, 0.3333333, 0.5]), decimal=5
    )
    assert_almost_equal(
        optimal_rate, np.array([0, 0, 0, 0.3333333, 0.5]), decimal=5
    )


def test_delinquency_curve__multilabel_exception():
    y_true = np.array([0, 0, 1, 1, 2])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8, 0.04])
    with pytest.raises(Exception) as exc_info:
        _, _ = delinquency_curve(y_true, y_scores)
    assert (
        str(exc_info.value)
        == "Only binary classification is supported. Provided [0 1 2]."
    )


def test_ranked_discrete_score_loss__two_examples_success():
    binary_y_true = np.array([1, 0])
    first_score = np.array([[0.1, 0.9], [0.9, 0.1]])
    rps = ranked_probability_score_loss(binary_y_true, first_score)
    assert rps < 0.01

    second_score = np.array([[1, 0], [0, 1]])
    rps = ranked_probability_score_loss(binary_y_true, second_score)
    assert rps == 1

    third_score = np.array([[0, 1], [1, 0]])
    rps = ranked_probability_score_loss(binary_y_true, third_score)
    assert rps == 0


def test_ranked_discrete_score_loss__four_examples_success():
    y_true = np.array([0, 0, 1, 0])
    y_prob_model_1 = np.array(
        [[0.7, 0.3], [0.85, 0.15], [0.3, 0.7], [0.9, 0.1]]
    )
    rps_model_1 = ranked_probability_score_loss(y_true, y_prob_model_1)
    y_prob_model_2 = np.array(
        [[0.6, 0.4], [0.85, 0.15], [0.3, 0.7], [0.9, 0.1]]
    )
    rps_model_2 = ranked_probability_score_loss(y_true, y_prob_model_2)
    assert rps_model_1 < rps_model_2

    y_prob_model_3 = np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.9, 0.1]])
    rps_model_3 = ranked_probability_score_loss(y_true, y_prob_model_3)
    assert rps_model_3 < rps_model_1


def test_ranked_discrete_score_loss__three_examples_three_classes_success():
    multi_y_true = np.array([2, 1, 0])
    first_score = np.array(
        [[0.08, 0.02, 0.9], [0.02, 0.9, 0.08], [0.9, 0.02, 0.08]]
    )

    rps = ranked_probability_score_loss(multi_y_true, first_score)
    assert rps < 0.01

    second_score = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    rps = ranked_probability_score_loss(multi_y_true, second_score)
    assert rps == 0

    third_score = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    rps = ranked_probability_score_loss(multi_y_true, third_score)
    assert rps > 0.8


def test_ranked_discrete_score_loss__three_examples_ordinal_success():
    multi_y_true = np.array([2, 1, 0])
    first_score = np.array(
        [[0.08, 0.02, 0.9], [0.02, 0.9, 0.08], [0.9, 0.02, 0.08]]
    )

    rps_first_model = ranked_probability_score_loss(multi_y_true, first_score)

    second_score = np.array(
        [[0.01, 0.09, 0.9], [0.02, 0.9, 0.08], [0.9, 0.02, 0.08]]
    )
    rps_second_model = ranked_probability_score_loss(
        multi_y_true, second_score
    )
    assert rps_second_model < rps_first_model


def test_ranked_discrete_score_loss__three_examples_ordinal_success_label():
    multi_y_true = np.array(["2", "1", "0"])
    label = ["0", "1", "2"]
    first_score = np.array(
        [[0.08, 0.02, 0.9], [0.02, 0.9, 0.08], [0.9, 0.02, 0.08]]
    )

    rps_first_model = ranked_probability_score_loss(
        multi_y_true, first_score, labels=label
    )

    second_score = np.array(
        [[0.01, 0.09, 0.9], [0.02, 0.9, 0.08], [0.9, 0.02, 0.08]]
    )
    rps_second_model = ranked_probability_score_loss(
        multi_y_true, second_score, labels=label
    )
    assert rps_second_model < rps_first_model


def test_ranked_discrete_score_loss__two_examples_success_sample_weight():
    binary_y_true = np.array([1, 0])
    first_score = np.array([[0.2, 0.8], [0.9, 0.1]])
    first_sample_weight = np.array([1, 1])
    rps = ranked_probability_score_loss(
        binary_y_true, first_score, sample_weight=first_sample_weight
    )
    assert rps < 0.03

    second_sample_weight = np.array([0.1, 1])
    rps = ranked_probability_score_loss(
        binary_y_true, first_score, sample_weight=second_sample_weight
    )
    assert rps < 0.015
