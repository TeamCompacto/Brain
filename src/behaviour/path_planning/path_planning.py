
import socket
from threading import Thread
from path_finding_algorithm.entities import Edge
from path_finding_algorithm.entities import Node
from path_finding_algorithm.a_star import AStar
from path_finding_algorithm.read_information import Reader
from src.templates.workerprocess import WorkerProcess

# if __name__ == '__main__':
#     g = Reader('Competition_track.graphml').G #The parameter is the name of the file containing the road map represented by a directed graph
#     nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in g.nodes(data=True)}
#     edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in g.edges(data=True)]
    
#     shortest_path, dist = AStar(nodes, edges, 150, 151).shortest_path
#     print(shortest_path, dist)

class PathFindingProcess(WorkerProcess):

    def __init__(self, inPs, outPs, daemon=True):
        super().__init__(inPs, outPs, daemon)

    def run(self):
        self._init_socket()
        super(PathFindingProcess, self).run()

    def _init_socket(self):
        self.port = 2023
        self.serverIp = '0.0.0.0'

        try:
            self.server_socket = socket.socket()
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket)
            self.server_socket.bind((self.serverIp, self.port))

            self.server_socket.listen(5)
            self.connection = self.server_socket.accept()[0].makefile('rb')

        except:
            print('Socket error')

    def _init_threads(self):
        pathTh = Thread(name='PathFindingThread', target=self._path_finding_thread)
        self.threads.append(pathTh)

    def _path_finding_thread(self):
        g = Reader('Competition_track.graphml').G #The parameter is the name of the file containing the road map represented by a directed graph
        nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in g.nodes(data=True)}
        edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in g.edges(data=True)]

        start_node = self.__get_start_node()

    def __get_start_node(self):
        start_coor = self.__get_coors()
        

    def __get_coors(self):
        raise NotImplementedError
