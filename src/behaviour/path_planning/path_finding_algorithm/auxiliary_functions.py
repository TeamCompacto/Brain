from math import sqrt


def distance(p1: tuple, p2: tuple):
    """Calculates the distance between two points

    Args:
        p1 (tuple): The first point: (x, y)
        p2 (tuple): The second point: (x, y)

    Returns:
        float: The distance
    """
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)