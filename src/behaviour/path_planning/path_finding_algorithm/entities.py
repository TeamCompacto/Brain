import math


class Node():
    def __init__(self, id, x, y) -> None:
        self.__id = id
        self.__x = x
        self.__y = y

    @property
    def id(self):
        return self.__id

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y
    
    def __eq__(self, other: object) -> bool:
        return get_clockwise_angle_and_distance(self) == get_clockwise_angle_and_distance(other)

    def __lt__(self, other: object) -> bool:
        return get_clockwise_angle_and_distance(self) < get_clockwise_angle_and_distance(other)
    
    def __str__(self) -> str:
        return f"{self.id}: x: {self.x}, y: {self.y}"
    
def get_clockwise_angle_and_distance(node: Node):
    refvec = [1, 0]
    lenvector = math.hypot(node.x, node.y)
    if lenvector == 0:
        return -math.pi, 0
    
    normalized = [node.x / lenvector, node.y / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
    diffprod = normalized[0] * refvec[1] - normalized[1] * refvec[0]
    angle = math.atan2(diffprod, dotprod)

    if angle < 0:
        return 2 * math.pi + angle, lenvector
    
    return angle, lenvector

class Edge():
    def __init__(self, source_id, destination_id, dotted) -> None:
        self.__source_id = source_id
        self.__destination_id = destination_id
        self.__dotted = dotted

    @property
    def src(self):
        return self.__source_id

    @property
    def dest(self):
        return self.__destination_id

    @property
    def dotted(self):
        return self.__dotted
    
class PriorityQueueElement():
    def __init__(self, id, minimum_ditance, relative_distance, prev_node) -> None:
        self.__id = id
        self.__min_dist = minimum_ditance
        self.__rel_dist = relative_distance
        self.__prev_node = prev_node

    @property
    def id(self):
        return self.__id
    
    @property
    def min_dist(self):
        return self.__min_dist

    @property
    def rel_dist(self):
        return self.__rel_dist
    
    @property
    def prev_node(self):
        return self.__prev_node
    
    def __eq__(self, other: object) -> bool:
        return self.rel_dist + self.min_dist == other.rel_dist + other.min_dist
    
    def __lt__(self, other: object) -> bool:
        return self.rel_dist + self.min_dist < other.rel_dist + other.min_dist