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