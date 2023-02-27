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
    
    # def __eq__(self, other: object) -> bool:
    #     return get_clockwise_angle_and_distance(self) == get_clockwise_angle_and_distance(other)

    # def __lt__(self, other: object) -> bool:
    #     return get_clockwise_angle_and_distance(self) < get_clockwise_angle_and_distance(other)
    
    def __str__(self) -> str:
        return f"{self.id}: x: {self.x}, y: {self.y}"
    
# def get_clockwise_angle_and_distance(node: Node):
#     refvec = [1, 0]
#     lenvector = math.hypot(node.x, node.y)
#     if lenvector == 0:
#         return -math.pi, 0
    
#     normalized = [node.x / lenvector, node.y / lenvector]
#     dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
#     diffprod = normalized[0] * refvec[1] - normalized[1] * refvec[0]
#     angle = math.atan2(dotprod, diffprod)

#     if angle < 0:
#         return 2 * math.pi + angle, lenvector
    
#     return angle, lenvector

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
    def __init__(self, id, minimum_distance, relative_distance, prev_node) -> None:
        """Elements for the priority queue for computing the shortest path

        Args:
            id (int): The id of the node
            minimum_distance (int): The minimum distance from the startnode
            relative_distance (int): The euqlidian distance from the endnode
            prev_node (int): The id of the previous node
        """
        self.__id = id
        self.__min_dist = minimum_distance
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
    
    def __str__(self) -> str:
        return f"{self.id}: min_dist: {self.min_dist}, rel_dist: {self.rel_dist}, prev: {self.prev_node}"
    

class DistanceNode():
    def __init__(self, id: int, min_distance: float, visited_events: dict) -> None:
        self.__id = id
        self.__min_dist = min_distance
        self.__visited_events = visited_events

    @property
    def id(self):
        return self.__id
    
    @property
    def min_dist(self):
        return self.__min_dist
    
    @property
    def visited_events(self):
        return self.__visited_events
    
    def __str__(self) -> str:
        return f"{self.id}: min_dist: {self.min_dist}, visited_events: {self.visited_events}"
    
