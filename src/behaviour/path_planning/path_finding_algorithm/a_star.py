from math import sqrt
from queue import PriorityQueue

from path_finding_algorithm.entities import PriorityQueueElement


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

        self.__shortest_path = self.__get_shortest_path()

    @property
    def shortest_path(self):
        return self.__shortest_path

    def __build_adjacent_list(self):
        """Builds the adjacent list of the graph with tuples: (node ID, distance between nodes)
        """
        adjacent_list = {}
        for node in self.__nodes:
            adjacent_list[node] = []

        for edge in self.__edges:
            adjacent_list[edge.src].append((edge.dest, self.distance((self.__nodes[edge.src].x, self.__nodes[edge.src].y), (self.__nodes[edge.dest].x, self.__nodes[edge.dest].y))))

        return adjacent_list

    def distance(self, p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def __get_shortest_path(self):
        min_distance = {}
        prev_node = {}

        pq = PriorityQueue()
        pq.put(PriorityQueueElement(self.__start_id, 0, 0, -1))

        while not pq.empty():
            curr_node = pq.get()
            min_distance[curr_node.id] = curr_node.min_dist
            prev_node[curr_node.id] = curr_node.prev_node
            if curr_node.id == self.__end_id:
                break

            # print(f"{curr_node.id} {curr_node.min_dist} {curr_node.rel_dist} {curr_node.prev_node}")

            for adjacent, curr_dist in self.__adjacent_list[curr_node.id]:
                if adjacent not in min_distance:
                    pq.put(PriorityQueueElement(adjacent, min_distance[curr_node.id] + curr_dist, self.distance((self.__nodes[adjacent].x, self.__nodes[adjacent].y), (self.__nodes[self.__end_id].x, self.__nodes[self.__end_id].y)), curr_node.id))

        # print(prev_node)

        shortest_path = []
        parse_node = self.__end_id
        while parse_node != -1:
            shortest_path.append(parse_node)
            parse_node = prev_node[parse_node]

        shortest_path.reverse()
        return shortest_path, min_distance[self.__end_id]
    

    