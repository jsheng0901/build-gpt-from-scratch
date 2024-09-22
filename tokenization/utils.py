import unicodedata


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts

    # iterate consecutive elements
    for pair in zip(ids, ids[1:]):
        # update pair frequency
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    # loop through all ids
    while i < len(ids):
        # if not at the very last position and the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            # add new idx to list
            new_ids.append(idx)
            # 2 step forward since we do pair merge
            i += 2
        # if not inside merge pair
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    # decode bytes string to string
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s
