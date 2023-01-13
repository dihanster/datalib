from ..metrics import discrete_ranked_probability_score, discrete_ranked_probability_loss

import numpy as np



def test_discrete_ranked_probability_score_curve__success_case_two_classes():

    binary_y_true = np.array([1, 0])
    first_score = np.array([0.9, 0.1])
    rps = discrete_ranked_probability_score(binary_y_true, first_score)
    assert rps < 0.01
    second_score = np.array([1, 0])
    rps = discrete_ranked_probability_score(binary_y_true, second_score)
    assert rps == 0
    third_score = np.array([0, 1])
    rps = discrete_ranked_probability_score(binary_y_true, third_score)
    assert rps == 1


def test_discrete_ranked_probability_score_curve__success_case_three_classes():

    multi_y_true = np.array([1, 0, 0])
    first_score = np.array([0.9, 0.05, 0.05])
    bias = len(multi_y_true) - 1
    rps = discrete_ranked_probability_score(multi_y_true, first_score, bias_correction=bias)
    assert rps < 0.01
    second_score = np.array([1, 0, 0])
    rps = discrete_ranked_probability_score(multi_y_true, second_score, bias_correction=bias)
    assert rps == 0
    third_score = np.array([0, 0, 1])
    rps = discrete_ranked_probability_score(multi_y_true, third_score, bias_correction=bias)
    assert rps == 1


    multi_y_true = np.array([0, 1, 0])
    first_score = np.array([0.05, 0.9, 0.05])
    rps = discrete_ranked_probability_score(multi_y_true, first_score, bias_correction=bias)
    assert rps < 0.01
    second_score = np.array([0, 1, 0])
    rps = discrete_ranked_probability_score(multi_y_true, second_score, bias_correction=bias)
    assert rps == 0
    third_score = np.array([1, 0, 0])
    rps = discrete_ranked_probability_score(multi_y_true, third_score, bias_correction=bias)
    assert rps == 0.5   


def test_discrete_ranked_probability_loss_binary_curve__success_case():

    y_true = np.array([0, 0, 1, 0])
    y_prob_model_1 = np.array([[0.7, 0.3], [0.85, 0.15], [0.3, 0.7], [0.9, 0.1]])
    rps_model_1 = discrete_ranked_probability_loss(y_true, y_prob_model_1)
    y_prob_model_2 = np.array([[0.6, 0.4], [0.85, 0.15], [0.3, 0.7], [0.9, 0.1]])
    rps_model_2 = discrete_ranked_probability_loss(y_true, y_prob_model_2)
    assert rps_model_1 < rps_model_2

    y_prob_model_3 = np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.9, 0.1]])
    rps_model_3 = discrete_ranked_probability_loss(y_true, y_prob_model_3)
    assert rps_model_3 < rps_model_1
 