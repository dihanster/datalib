from datalib.utils import all_equal_elements


def test__all_equal_elements():
    all_equal_elements_list = 10 * [0]
    not_all_equal_elements_list = 10 * [0] + 5 * [1]
    unique_element_list = [0]
    empty_list = []
    assert all_equal_elements(all_equal_elements_list)
    assert not all_equal_elements(not_all_equal_elements_list)
    assert all_equal_elements(unique_element_list)
    assert all_equal_elements(empty_list)
