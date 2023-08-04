from datalib.utils import all_equal_elements


def test__all_equal_elements():
    all_equal_elements_list = 10 * [0]
    not_all_equal_elements_list = 10 * [0] + 5 * [1]
    assert all_equal_elements(all_equal_elements_list)
    assert not all_equal_elements(not_all_equal_elements_list)
