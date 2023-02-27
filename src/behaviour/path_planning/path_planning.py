
import math
from random import uniform
import socket
from threading import Thread
from path_finding_algorithm.entities import Edge
from path_finding_algorithm.entities import Node
from path_finding_algorithm.a_star import AStar
from path_finding_algorithm.read_information import Reader
from path_finding_algorithm.auxiliary_functions import distance
from path_finding_algorithm.entities import DistanceNode
# from src.templates.workerprocess import WorkerProcess

# if __name__ == '__main__':
#     g = Reader('Competition_track.graphml').G #The parameter is the name of the file containing the road map represented by a directed graph
#     nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in g.nodes(data=True)}
#     edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in g.edges(data=True)]
    
#     shortest_path, dist = AStar(nodes, edges, 150, 151).shortest_path
#     print(shortest_path, dist)

# class PathFindingProcess(WorkerProcess):

#     def __init__(self, inPs, outPs, daemon=True):
#         super().__init__(inPs, outPs, daemon)

#     def run(self):
#         self._init_socket()
#         super(PathFindingProcess, self).run()

#     def _init_socket(self):
#         self.port = 2023
#         self.serverIp = '0.0.0.0'

#         try:
#             self.server_socket = socket.socket()
#             self.server_socket.setsockopt(socket.SOL_SOCKET, socket)
#             self.server_socket.bind((self.serverIp, self.port))

#             self.server_socket.listen(5)
#             self.connection = self.server_socket.accept()[0].makefile('rb')

#         except:
#             print('Socket error')

#     def _init_threads(self):
#         pathTh = Thread(name='PathFindingThread', target=self._path_finding_thread)
#         self.threads.append(pathTh)

#     def _path_finding_thread(self):
#         g = Reader('Competition_track.graphml').G #The parameter is the name of the file containing the road map represented by a directed graph
#         nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in g.nodes(data=True)}
#         edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in g.edges(data=True)]
#         nodes = sorted(nodes)
#         for node in nodes:
#             print(node)

#         start_node = self.__get_start_node()

#     def __get_start_node(self):
#         start_coor = self.__get_coors()


#     def __get_coors(self):
#         raise NotImplementedError
    
class PathFindingProcess():

    def __init__(self, graph_map_name):
        self.__graph_map_name = graph_map_name
        self.__g = Reader(self.__graph_map_name).G #The parameter is the name of the file containing the road map represented by a directed graph
        self.__nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in self.__g.nodes(data=True)}
        self.__edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in self.__g.edges(data=True)]
        
        self._path_finding_thread()

    def _path_finding_thread(self):
        event_nodes = {"roundabout": [230, 301, 342], "highway": [311, 347], "parking": [177, 162], "road_closed_stand": [7, 8], 
                       "one_way_one_lane": [426], "finish_line": [85]}
        
        start_node = self.__get_start_node()

        self.__find_best_traversing(event_nodes, start_node)

    def __get_start_node(self):
        start_coor = self.__get_coors()
        return self.__find_closest_node(start_coor)    

    def __get_coors(self):
        return (round(uniform(0, 14), 2), round(uniform(0, 14), 2))
    
    def __find_closest_node(self, start_coor):
        sol = Node(-1, 0, 0)
        min_dist = -1
        for node in self.__nodes.values():
            if sol.id == -1 or distance(start_coor, (node.x, node.y)) < min_dist:
                sol = node
                min_dist = distance(start_coor, (node.x, node.y))

        return sol
    
    def __find_best_traversing(self, event_nodes: dict, start_node: Node):
        print(start_node)
        distances_with_order_dp = [{}]
        for order_num in range(1, len(event_nodes)):
            
            curr_distances = {}
            for curr_event in event_nodes:
                curr_distances[curr_event] = {} 
                for curr_event_node in event_nodes[curr_event]:
                    if order_num == 1:
                        path, dist = AStar(self.__nodes, self.__edges, start_node.id, curr_event_node).shortest_path
                        visited_events = {event: False for event in event_nodes}
                        visited_events[curr_event] = True
                        curr_distances[curr_event][curr_event_node] = DistanceNode(curr_event_node, dist, visited_events)
                    # else:
                    #     for prev_event in event_nodes:
                    #         if prev_event == curr_event:
                    #             continue
                    #         for prev_event_node in event_nodes[prev_event]:

                                



            distances_with_order_dp.append(curr_distances)



        print(distances_with_order_dp)              
                    
    # def big_dict_printer(self, big_list):
    #     for big_dict in big_list:
    #         print("{", end="")
    #         for event in big_dict:
    #             print(f"{event}: ", "{", end="")
    #             for node in big_dict[event]:
    #                 print(f"{node}: ", end="")
    #                 for elem in big_dict[event].values():
    #                     print(elem, end=" ")
    #             print("}", end="")
    #         print("}", end="")

if __name__ == "__main__":
    path_finding = PathFindingProcess("Competition_track.graphml")

    start_coor = (round(uniform(0, 14), 2), round(uniform(0, 14), 2))



