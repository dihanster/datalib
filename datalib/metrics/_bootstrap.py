import numpy as np
from sklearn.utils import check_array, check_random_state

from ..utils import all_equal_size


def bootstrap_metric(metric, *args, random_state=42, n_bootstrap=20, **kwargs):
    """TODO"""
    args = [check_array(v, ensure_2d=False, dtype=None) for v in args]
    kwargs = {
        k: check_array(v, ensure_2d=False, dtype=None)
        for k, v in kwargs.items()
    }

    if not all_equal_size(
        [len(array) for array in list(kwargs.values()) + list(args)]
    ):
        msg_error_size = "All the elements to be bootstraped (*args and **kwargs) are not equal in length."
        raise ValueError(msg_error_size)

    size = len((list(kwargs.values()) + args)[0])

    # Run usual metric checks.
    metric(*args, **kwargs)

    rng = check_random_state(random_state)
    random_states = rng.randint(
        low=0, high=2**32 - 1, size=n_bootstrap, dtype=np.int64
    )

    metric_list = []
    for rs in random_states:
        idx = check_random_state(rs).choice(a=size, size=size, replace=True)
        bootstrap_args = [v[idx] for v in args]
        bootstrap_kwargs = {k: v[idx] for k, v in kwargs.items()}
        metric_round = metric(*bootstrap_args, **bootstrap_kwargs)
        metric_list.append(metric_round)

    return metric_list
