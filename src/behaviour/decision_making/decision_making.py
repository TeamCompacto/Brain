import socket
from threading import Thread
from src.communication.enums.obstacles import Obstacles
from src.templates.workerprocess import WorkerProcess

class DecisionMakingProcess(WorkerProcess):
    
    def __init__(self, inPs, outPs, daemon=True):
        super().__init__(inPs, outPs, daemon)

    def run(self):
        self._init_socket()
        super(DecisionMakingProcess, self).run()

    def _init_socket(self):
        self.port = 2023
        self.serverIp = '0.0.0.0'

        try:
            self.server_socket = socket.socket()
            self.server_socket.setsocketopt(socket.SOL_SOCKET, socket)
            self.server_socket.bind((self.serverIp, self.port))

            self.server_socket.listen(5)
            self.connection = self.server_socket.accept()[0].makefile('rb')

        except:
            print('Socket error')

    def _init_threads(self):
        decTh = Thread(name='DecisionMakingThread', target=self._decision_making_thread)
        self.threads.append(decTh)

    def _decision_making_thread(self):
        obstacles = dict()
        # TODO: get obstacles

        for obstacle in obstacles:

            if obstacle.id == Obstacles.STOP_SIGN:
                
                if obstacle.x2 - obstacle.x1 >= 250:

                    pass


            else:
                print("Invalid id")
