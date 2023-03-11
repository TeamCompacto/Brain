import socket
from threading import Thread
from src.communication.enums.obstacles import Obstacles
from src.templates.workerprocess import WorkerProcess
import time

class DecisionMakingProcess(WorkerProcess):
    
    def __init__(self, inPs, outPs):
        super().__init__(inPs, outPs)
        self.current_state = "BASE"
        self.current_speed = 0.0
        self.current_steering_angle = 0.0

    def run(self):
        super(DecisionMakingProcess, self).run()

    def _init_threads(self):
        decTh = Thread(name='DecisionMakingThread', target=self._decision_making_thread, args=([self.inPs[0], self.inPs[1]], self.outPs[0]))
        self.threads.append(decTh)

    def _decision_making_thread(self, inPs, outPs):
        print("Decision Making Started")
        outPs.send({'action': '3', 'brake (steerAngle)': 0.0} )
        outPs.send({'action': '1', 'speed': 0.14} )
        time.sleep(0.1)
        outPs.send({'action': '1', 'speed': 0.09} )
        try:
            count = 0
            
            while True:
                [deviation] = inPs[0].recv()
                res = inPs[1].recv()

                print("Decision: starting to process data", str(count), " at ", str(time.ctime()))
                

                if self.current_state == "BASE":
                    outPs.send({'action': '3', 'brake (steerAngle)': self.current_steering_angle} )
                    time.sleep(1)
                    outPs.send({'action': '1', 'speed': 0.14} )
                    time.sleep(0.1)
                    outPs.send({'action': '1', 'speed': 0.09} )

                print("Received deviation:", deviation)

                for sign in res:
                    print("Detected sign with id: ", sign['class_id'])
                    if sign['class_id'] == 0:
                        print("stopping")
                        print(time.time())
                        outPs.send({'action': '3', 'brake (steerAngle)': 0.0} )
                        time.sleep(3)
                        outPs.send({'action': '1', 'speed': 0.12} )
                        time.sleep(0.2)
                        outPs.send({'action': '1', 'speed': 0.09} )
                        time.sleep(0.1)


                if deviation == 700:
                    self.current_steering_angle = 20.0
                    outPs.send({'action': '2', 'steerAngle': 20.0} )
                elif deviation == -700:
                    self.current_steering_angle = -20.0
                    outPs.send({'action': '2', 'steerAngle': -20.0} )
                elif deviation < -100:
                    self.current_steering_angle = -10.0
                    outPs.send({'action': '2', 'steerAngle': -10.0} )
                    
                elif deviation > 100:
                    self.current_steering_angle = 10.0
                    outPs.send({'action': '2', 'steerAngle': 10.0} )
                else:
                    self.current_steering_angle = 0.0
                    outPs.send({'action': '2', 'steerAngle': 0.0} )

                print("Decision: finished processing data", str(count), " at ", str(time.ctime()))
                count += 1
                

        except KeyboardInterrupt:
            outPs.send({'action': '3', 'brake (steerAngle)': 0.0} )

        # obstacles = list()
        # # TODO: get obstacles

        # for obstacle in obstacles:

        #     if obstacle.id == Obstacles.STOP_SIGN:
                
        #         if obstacle.x2 - obstacle.x1 >= 250:

        #             pass


        #     else:
        #         print("Invalid id")
