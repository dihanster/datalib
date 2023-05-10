# Create tests...


def update_dic(dic1, dict2):
    return (lambda d: d.update(dict2) or d)(dic1.copy())


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
