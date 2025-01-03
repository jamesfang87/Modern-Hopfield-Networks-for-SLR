import numpy as np


def downsample_sign(sign: np.ndarray, new_length: int):
    """
    Decreases the length (number of frames) of a sign to 'new_length'. We sample frames
    from the original sign based on an increment calculated by dividing new_length by
    the original length.

    :param sign: the original sign of length, which is longer than new_length
    :param new_length: the length to project 'sign' to
    :return: the padded sign of length 'new_length'
    """

    # the projected length should be less than the current length
    assert new_length < len(sign)
    # sign should be 3-dimensional: T x landmarks x (3 or 2 based on spatial dimensions)
    assert len(sign.shape) == 3

    new_sign = np.zeros((new_length, sign.shape[1], sign.shape[2]))

    # sample frames from the original sign based on the difference
    increment = new_length / len(sign)
    current_frame = 0  # the frame index for the original sign
    for i in range(new_length):
        new_sign[i] = sign[int(current_frame)]
        current_frame += increment

    return new_sign


def upsample_sign(sign: np.ndarray, new_length: int):
    """
    Increases the length (number of frames) of a sign to 'new_length'.
    The sign is padded with duplicates of the first frame at the start and duplicates
    of the second frame at the end so that the actual sign is "centered" in the new length.

    :param sign: the original sign of length, which is shorter than new_length
    :param new_length: the length to project 'sign' to
    :return: the padded sign of length 'new_length'
    """

    # the projected length should be greater than the current length
    assert new_length > len(sign)

    # sign should be 3-dimensional: T x landmarks x (3 or 2 based on spatial dimensions)
    assert len(sign.shape) == 3

    difference: int = new_length - len(sign)
    before_sign: int = difference // 2  # num of frames to pad at the start
    after_sign: int = difference - before_sign  # num of frames to pad at the end

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


def normalize_pose_landmarks():
    """
    Normalizes pose landmarks based on the size of the signing space of the person
    :return:
    """
    raise NotImplementedError
