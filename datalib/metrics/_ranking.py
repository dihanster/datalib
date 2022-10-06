
def ranked_probability_score(y_true, y_proba):
    """Ranked Probability Score.

    The RPS is used to quantify the performance of probabilistic prediction systems. It compares
    the cumulative density function of a probabilistic forecast with a ground truth, condined by a 
    number of levels(Categories). The BS is the two-level case of RPS. 
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Correct labels for given dataset.
    y_proba : array, shape = [n_samples]
        Predicted probability scores for the given dataset.
    Returns
    -------
    approval_rate: array.
        An array containing the approval rates used to compute the default_rate curve.
    default_rate: array.
        An array containing the default rates values for the approval rates provided in approval_rate.
    optimal_rate: array.
        An array containing the optimal default rates for a perfect model.
    """