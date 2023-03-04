import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight


def bootstrap_metric(
        y_true,
        y_pred,
        metric,
        n_bootstrap=20,
        sample_weight=None,
        random_state=None,
        **kwargs):
    """TO DO
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_pred = check_array(y_pred, ensure_2d=False)
    sample_weight = _check_sample_weight(sample_weight)

    # runs metrics checks
    metric(y_true, y_pred, sample_weight=sample_weight, **kwargs)

    rng = check_random_state(random_state)
    random_states = rng.randint(
            low=0, high=2**32 - 1, size=n_bootstrap, dtype=np.int64
        )
    metric_list = []
    for rs in random_states:
        idx = check_random_state(rs).choice(len(y_true), len(y_true), replace=True)
        metric_round = metric(
            y_true[idx], y_pred[idx], sample_weight=sample_weight[idx], **kwargs
        )
        metric_list.append(metric_round)
    
    return np.mean(metric_list), np.std(metric_list)