from math import sqrt


class AStar():
    def __init__(self, nodes, edges, source_node_id, destination_node_id) -> None:
        self.__nodes = nodes
        self.__edges = edges
        # for node in self.__nodes.values():
        #     print(node.x, node.y)
        # for edge in self.__edges:
        #     print(f"{edge.src} {edge.dest}")
        self.__start_id = source_node_id
        self.__end_id = destination_node_id

        self.__adjacent_list = self.__build_adjacent_list()

    def __build_adjacent_list(self):
        """Builds the adjacent list of the graph with tuples: (node ID, distance between nodes)
        """
        adjacent_list = {}
        for edge in self.__edges:
            if edge.src not in adjacent_list:
                adjacent_list[edge.src] = []

            print(adjacent_list[edge.src])
            print((self.__nodes[edge.src].x))
            adjacent_list[edge.src].append((edge.dest, self.distance((self.__nodes[edge.src].x, self.__nodes[edge.src].y), (self.__nodes[edge.dest].x, self.__nodes[edge.dest].y))))

        # print(adjacent_list)

    def distance(self, p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    