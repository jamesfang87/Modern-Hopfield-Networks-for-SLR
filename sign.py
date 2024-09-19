import numpy as np
import math


class SignInfo:
    """
    this class holds all features which are used to distinguish between signs
    features are only for one instance/example of a sign

    features:
    num_hands: the number of hands in the sign (either 1 or 2)
    dom_hand_pos: np array of hand positions (center) of the dominant hand given in (x, y)
    ndom_hand_pos: np array of hand positions (center) of the non-dominant hand given in (x, y)
    dom_hand_motion: np array of unit vectors describing the change in hand positions for dominant hand
    ndom_hand_motion: np array unit vectors describing the change in hand positions for non-dominant hand
    relative_pos: np array of distance between dominant and non-dominant hands given in (x, y)
    relative_pos_change: np array of unit vectors describing the change in relative_pos
    handshapes: list of handshapes

    this class is passed into the Dictionary class's sign_to_word method
    """

    def __init__(self, num_hands: int, handface=None):
        """
        :param num_hands: the number of hands in the sign (either 1 or 2)
        :param handface: the matlab file representing the sign
        :param raw_vid_info: the class holding information of the sign from the camera
        """
        self.num_hands = num_hands
        self.init_from_matlab(handface)
        

    def init_from_matlab(self, handface):
        # raw data for the bounding box of each hand + face
        # given in [left_col, top_row, width, height]

        # where T is the number of frames in the sign:
        # array of size (T x 4) representing the bounding box of the dominant hand
        dom = handface[0]
        # array of size (T x 4) representing the bounding box of the non-dominant hand
        ndom = handface[1]
        # array of size (1 x 4) representing the bounding box of the face
        face = handface[2]

        # face center is taken as (0, 0)
        face_center = get_centers(face)

        # scaling distance is based on the size of the face bounding box
        widths = face[:, 2]
        heights = face[:, 3]
        scaling_dist: float = (widths + heights) / 2

        # np array of hand positions (center) of the dominant hand given in (x, y)
        self.dom_hand_pos = get_centers(dom)
        self.dom_hand_pos = normalize_length(self.dom_hand_pos, 20)
        self.dom_hand_pos = ((self.dom_hand_pos - face_center) / scaling_dist) * 2.0

        # np array of unit vectors describing the change in hand positions for dominant hand
        unnormalized_dom_hand_motion = np.diff(self.dom_hand_pos, axis=0)
        dom_hand_norms = np.linalg.norm(unnormalized_dom_hand_motion, axis=1)[:, np.newaxis]
        self.dom_hand_motion = 1.24 * unnormalized_dom_hand_motion / dom_hand_norms
        self.dom_hand_motion[np.isnan(self.dom_hand_motion)] = 0

        if self.num_hands == 2:
            # np array of hand positions (center) of the non-dominant hand given in (x, y)
            self.ndom_hand_pos = get_centers(ndom)
            self.ndom_hand_pos = normalize_length(self.ndom_hand_pos, 20)
            self.ndom_hand_pos = ((self.ndom_hand_pos - face_center) / scaling_dist) * 1.4

            # np array unit vectors describing the change in hand positions for non-dominant hand
            unnormalized_ndom_hand_motion = np.diff(self.ndom_hand_pos, axis=0)
            ndom_hand_norms = np.linalg.norm(unnormalized_ndom_hand_motion, axis=1)[:, np.newaxis]
            self.ndom_hand_motion = 1.24 * unnormalized_ndom_hand_motion / ndom_hand_norms
            self.ndom_hand_motion[np.isnan(self.ndom_hand_motion)] = 0

            # np array of distance between dominant and non-dominant hands given in (x, y)
            self.relative_pos = (self.dom_hand_pos - self.ndom_hand_pos) * 1.2

            # np array of unit vectors describing the change in relative_pos
            unnormalized_relative_pos_change = np.diff(self.relative_pos, axis=0)
            relative_pos_change_norm = np.linalg.norm(unnormalized_relative_pos_change, axis=1)[:, np.newaxis]
            self.relative_pos_change = 0.4 * (unnormalized_relative_pos_change / relative_pos_change_norm)
            self.relative_pos_change[np.isnan(self.relative_pos_change)] = 0
        else:
            # initialize to 0 for all 1-handed signs
            self.ndom_hand_pos = 0
            self.ndom_hand_motion = 0

            self.relative_pos = 0
            self.relative_pos_change = 0

        # the following features are not compared for signs with num_hands == 1:
        # ndom_hand_pos
        # ndom_hand_motion
        # relative_pos
        # relative_pos_change

        # there are also instance variables which consist of other features stacked together
        # they are there to speed up classification using DTW

        # base_features contains dom_hand_pos and dom_hand_motion
        # it will always be present no matter the handedness of the sign
        self.base_features = np.concatenate((
            self.dom_hand_pos[:-1],
            self.dom_hand_motion
        ), axis=1)

        if self.num_hands == 2:
            # features which depend on the sign being 2-handed
            self.two_handed_features = np.concatenate((
                self.ndom_hand_pos[:-1],
                self.ndom_hand_motion,
                self.relative_pos[:-1],
                self.relative_pos_change
            ), axis=1)

            # all_features is base_features + two_handed_features if sign is 2-handed
            self.all_features = np.concatenate((self.base_features, self.two_handed_features), axis=1)
        else:
            # two_handed_features set to 0
            self.two_handed_features = 0
            # otherwise it is just base_features
            self.all_features = self.base_features


def normalize_length(feature: np.ndarray, to_length: int) -> np.ndarray:
    length = len(feature)

    a = np.linspace(0, length - 2, to_length)
    normalized_feature = np.zeros((to_length, 2))
    for i in range(to_length):
        x1 = math.floor(a[i])
        x2 = x1 + 1

        y1 = feature[x1]
        y2 = feature[x2]

        slope = (y2 - y1) / (x2 - x1)
        delta_x = a[i] - x1
        normalized_feature[i] = slope * delta_x + y1

    return normalized_feature


def get_centers(positions):
    positions = positions.astype(float)
    lefts = positions[:, 0]
    tops = positions[:, 1]
    widths = positions[:, 2]
    heights = positions[:, 3]

    rights = lefts + widths - 1
    bottoms = tops + heights - 1

    (frames, _) = positions.shape
    result = np.zeros((frames, 2), dtype=float)
    result[:, 0] = (lefts + rights) / 2
    result[:, 1] = (tops + bottoms) / 2

    return result