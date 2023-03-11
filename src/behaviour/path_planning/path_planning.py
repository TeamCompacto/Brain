
from copy import deepcopy
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
                       "one_way_one_lane": [426]}
        finish_line = 85
        
        start_node = self.__get_start_node()

        self.__find_best_traversing_backtracking(event_nodes, start_node, finish_line)

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
    
    # def __find_best_traversing(self, event_nodes: dict, start_node: Node, finish_line: int):
        # print(start_node)
        # distances_with_order_dp = [{}]
        # for order_num in range(1, len(event_nodes) + 1):
        #     print(order_num, ":")
        #     curr_distances = {}
        #     for curr_event in event_nodes:
        #         print(curr_event, ":")
        #         curr_distances[curr_event] = {} 
        #         for curr_event_node in event_nodes[curr_event]:
        #             # print(curr_event_node, ": ", end="")
        #             if order_num == 1:
        #                 path, dist = AStar(self.__nodes, self.__edges, start_node.id, curr_event_node).shortest_path
        #                 visited_events = {event: False for event in event_nodes}
        #                 visited_events[curr_event] = True
        #                 curr_distances[curr_event][curr_event_node] = DistanceNode(curr_event_node, dist, visited_events, path)
        #                 print(curr_distances[curr_event][curr_event_node])
        #             else:
        #                 min_dist = math.inf; min_path = []
        #                 opt_event = -1; opt_node = -1
        #                 for prev_event in event_nodes:
        #                     if prev_event == curr_event:
        #                         continue
        #                     for prev_event_node in event_nodes[prev_event]:
        #                         if distances_with_order_dp[order_num - 1][prev_event][prev_event_node].visited_events[curr_event] == True:
        #                             continue
        #                         path, dist = AStar(self.__nodes, self.__edges, prev_event_node, curr_event_node).shortest_path
        #                         dist = dist + distances_with_order_dp[order_num - 1][prev_event][prev_event_node].min_dist
        #                         if dist < min_dist:
        #                             min_dist = dist
        #                             min_path = path
        #                             opt_event = prev_event
        #                             opt_node = prev_event_node

        #                 visited_events = deepcopy(distances_with_order_dp[order_num - 1][opt_event][opt_node].visited_events)
        #                 visited_events[curr_event] = True
        #                 curr_distances[curr_event][curr_event_node] = DistanceNode(curr_event_node, min_dist, visited_events, min_path)
        #                 print(curr_distances[curr_event][curr_event_node])
        #     distances_with_order_dp.append(curr_distances)
        pass

    def __find_best_traversing_backtracking(self, event_nodes: dict, start_node: Node, finish_line: int):
        visited_events = {event: False for event in event_nodes}
        tracking_list = [(start_node.id, 0)]
        self.optimal_list = []
        self.min_end_distance = math.inf
        self.__recursive_backtracking(tracking_list, event_nodes, visited_events, finish_line)
        print(self.optimal_list)

    def __recursive_backtracking(self, tracking_list, event_nodes, visited_events, finish_line):
        
        if len(tracking_list) == len(event_nodes) + 1:
            # print(self.optimal_list)
            end_distance = tracking_list[-1][1] + AStar(self.__nodes, self.__edges, tracking_list[-1][0], finish_line).shortest_path[1]
            if end_distance < self.min_end_distance:
                self.min_end_distance = end_distance
                self.optimal_list = deepcopy(tracking_list)
                # print(self.optimal_list)
            return

        for event in event_nodes:
            if visited_events[event] == False:
                visited_events[event] = True
                for event_node in event_nodes[event]:
                    tracking_list.append((event_node, tracking_list[-1][1] + AStar(self.__nodes, self.__edges, tracking_list[-1][0], event_node).shortest_path[1]))
        
                    self.__recursive_backtracking(tracking_list, event_nodes, visited_events, finish_line)

                    tracking_list.pop()
                visited_events[event] = False
                    
                

if __name__ == "__main__":
    path_finding = PathFindingProcess("Competition_track.graphml")

    start_coor = (round(uniform(0, 14), 2), round(uniform(0, 14), 2))



