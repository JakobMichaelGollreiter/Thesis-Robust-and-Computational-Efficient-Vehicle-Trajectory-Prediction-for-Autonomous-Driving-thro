# third-party imports
import numpy as np


def calculate_vtr_angle(vtr1: np.array, vtr2: np.array):
    """
    Return the angle between two vectors

    Args:
        vtr1 (np.array): First vector.
        vtr2 (np.array): Second vector.

    Returns:
        float, float: Angle in radians and degree format
    """
    cosTh = np.dot(vtr1, vtr2)
    sinTh = np.cross(vtr1, vtr2)
    angle_rad = np.arctan2(sinTh, cosTh)
    angle_deg = np.rad2deg(angle_rad)
    return angle_rad, angle_deg


def calculate_distance(pos1: np.array, pos2: np.array):
    """
    Return the euclidean distance between 2 points.

    Args:
        pos1 (np.array): First point.
        pos2 (np.array): Second point.

    Returns:
        float: Distance between point 1 and point 2.
    """
    return np.linalg.norm(pos1 - pos2)


def increase_path_resolution(lanelet_path: np.array):
    """
    Return the upsampled path based on the given vertices.

    Args:
        lanelet_path (np.array): center line or border pat hof the lanelet

    Returns:
        np.array: Array containing the upsampled path.
    """
    refined_path = []
    for index in range(len(lanelet_path) - 1):

        refined_path.append(lanelet_path[index])
        p1 = lanelet_path[index]
        p2 = lanelet_path[index + 1]

        midpoint = [0.5*(p1[0] + p2[0]), 0.5*(p1[1] + p2[1])]
        refined_path.append(np.array(midpoint))

    refined_path.append(lanelet_path[len(lanelet_path) - 1])
    return np.array(refined_path)
