import numpy as np


def __rotate(point: np.ndarray, pivot: np.ndarray, angle: float):
    """
    Rotates 'point' around 'pivot' by 'angle' radians
    :param point: np.ndarray of shape (2, 1) which is the point to rotate
    :param pivot: np.ndarray of shape (2, 1) to rotate 'point' around
    :param angle: the angle in radians to rotate by counterclockwise
    :return: the rotated point (of same type and shape)
    """
    px, py = point
    cx, cy = pivot

    # subtract the pivot so that it is at the origin
    translated_x = px - cx
    translated_y = py - cy

    # Apply rotation
    rotated_x = (translated_x * np.cos(angle)) - (translated_y * np.sin(angle))
    rotated_y = (translated_x * np.sin(angle)) + (translated_y * np.cos(angle))

    # Translate back to center
    new_x = rotated_x + cx
    new_y = rotated_y + cy

    return np.ndarray([new_x, new_y])


def horizontal_flip(sign: np.ndarray, probability: float):
    num = np.random.rand()
    if num > probability:
        return sign * -1
    else:
        return sign


def perspective_transform(probability: float, sign: np.ndarray):
    raise NotImplementedError


def joint_rotation(probability: float, sign: np.ndarray):
    raise NotImplementedError
