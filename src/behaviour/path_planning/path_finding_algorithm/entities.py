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
    def source_id(self):
        return self.__source_id

    @property
    def destination_id(self):
        return self.__destination_id

    @property
    def dotted(self):
        return self.__dotted