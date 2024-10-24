import numpy as np


def change_length(sign: np.ndarray, new_length: int):
    """
    Changes the length (number of frames) of a sign so that all signs have the same length.
    The sign is padded with duplicates of the first frame at the start and duplicates
    of the second frame at the end so that the actual sign is "centered" in the new length.

    :param sign: the original sign of length shorter than new_length
    :param new_length: the length to project 'sign' to
    :return: the padded sign of length 'new_length'
    """
    # the projected length should be greater than the current length
    assert new_length >= len(sign)

    difference: int = new_length - len(sign)
    before_sign: int = difference // 2  # num of frames to pad at the start
    after_sign: int = difference - before_sign  # num of frames to pad at the end

    # sign should be 3-dimensional: T x landmarks x (3 or 2 based on spatial dimensions)
    assert len(sign.shape) == 3

    new_sign = np.zeros((new_length, sign.shape[1], sign.shape[2]))

    # fill in duplicates of the first frame at the start
    for i in range(before_sign):
        new_sign[i] = sign[0]

    # fill in the actual sign
    for i in range(len(sign)):
        new_sign[i + before_sign] = sign[i]

    # fill in duplicates of the last frame at the end
    for i in range(after_sign):
        new_sign[i + before_sign + len(sign)] = sign[-1]

    return new_sign
