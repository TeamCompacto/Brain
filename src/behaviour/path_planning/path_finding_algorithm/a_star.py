class AStar():
    def __init__(self, nodes, edges, source_node_id, destination_node_id) -> None:
        self.__nodes = nodes
        self.__edges = edges
        self.__start_id = source_node_id
        self.__end_id = destination_node_id

        self.__build_adjacent_list()

    def __build_adjacent_list(self):
        pass

