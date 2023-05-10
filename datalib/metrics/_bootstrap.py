import numpy as np
from sklearn.utils import check_array, check_random_state

# TODO: change name ._.
from ..utils.nilvo import update_dic, all_equal


def bootstrap_metric(
    metric, kwargs_to_sample, *, random_state=42, n_bootstrap=20, **kwargs
):
    """TODO"""
    to_sample_dict = {
        k: check_array(v, ensure_2d=False, dtype=None)
        for k, v in kwargs.items()
        if k in kwargs_to_sample
    }
    not_to_sample_dict = {
        k: v for k, v in kwargs.items() if k not in kwargs_to_sample
    }

    # TODO: raise an error when this does not happen
    assert all_equal([len(array) for array in to_sample_dict.values()])
    size = len(list(to_sample_dict.values())[0])

    # Run usual sklearn checks.
    metric(**kwargs)

    rng = check_random_state(random_state)
    random_states = rng.randint(
        low=0, high=2**32 - 1, size=n_bootstrap, dtype=np.int64
    )

    metric_list = []
    for rs in random_states:
        idx = check_random_state(rs).choice(a=size, size=size, replace=True)
        bootstrap_dict = {k: v[idx] for k, v in to_sample_dict.items()}
        metric_round = metric(**update_dic(bootstrap_dict, not_to_sample_dict))
        metric_list.append(metric_round)

    return metric_list
