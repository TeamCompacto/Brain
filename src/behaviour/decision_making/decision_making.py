import socket
from threading import Thread
from src.communication.enums.obstacles import Obstacles
from src.templates.workerprocess import WorkerProcess

class DecisionMakingProcess(WorkerProcess):
    
    def __init__(self, inPs, outPs):
        super().__init__(inPs, outPs)

    def run(self):
        super(DecisionMakingProcess, self).run()

    def _init_threads(self):
        decTh = Thread(name='DecisionMakingThread', target=self._decision_making_thread, args=(self.inPs[0], self.outPs[0]))
        self.threads.append(decTh)

    def _decision_making_thread(self, inPs, outPs):
        print("Decision Making Started")
        outPs.send({'action': '1', 'speed': 0.12}  )
        try:
            while True:
                [deviation] = inPs.recv()
                print(type(deviation))
                print("Received deviation:", deviation)
                if deviation < -100:
                    print("sending turn left")
                    outPs.send({'action': '2', 'steerAngle': -10}  )
                if deviation > 100:
                    print("sending turn left")
                    outPs.send({'action': '2', 'steerAngle': 10}  )
        finally:
            outPs.send({'action': '3', 'brake (steerAngle)': 0.0} )



        # obstacles = list()
        # # TODO: get obstacles

        # for obstacle in obstacles:

        #     if obstacle.id == Obstacles.STOP_SIGN:
                
        #         if obstacle.x2 - obstacle.x1 >= 250:

        #             pass


        #     else:
        #         print("Invalid id")
