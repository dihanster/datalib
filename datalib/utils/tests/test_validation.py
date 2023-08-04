from datalib.utils import all_equal_size


def test__all_equal_size():
    all_equal_size_list = [10 * [0], 10 * [1]]
    not_all_equal_size_list = [10 * [0], 5 * [1]]
    assert all_equal_size(all_equal_size_list)
    assert not all_equal_size(not_all_equal_size_list)
