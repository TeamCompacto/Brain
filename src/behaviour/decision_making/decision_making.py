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
        decTh = Thread(name='DecisionMakingThread', target=self._decision_making_thread, args=([self.inPs[0], self.inPs[1]], self.outPs))
        self.threads.append(decTh)

    def _decision_making_thread(self, inPs, outPs):
        print("Decision Making Started")

        
        
        
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0} )
        outPs[0].send({'action': '1', 'speed': 0.14} )
        time.sleep(0.1)
        outPs[0].send({'action': '1', 'speed': 0.09} )
        try:
            count = 0
            outPs[1].send("I'm ready" + str(count))
            outPs[2].send("I'm ready" + str(count))
            
            while True:
                [deviation] = inPs[0].recv()
                res = inPs[1].recv()
                

                print("Decision: starting to process data", str(count), " at ", str(time.ctime()))

                

                for sign in res:
                    print("Detected sign with id: ", sign['class_id'])
                    if sign['class_id'] == 0:
                        print("stopping")
                        
                        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0} )
                        time.sleep(3)
                        outPs[0].send({'action': '1', 'speed': 0.12} )
                        time.sleep(0.2)
                        outPs[0].send({'action': '1', 'speed': 0.09} )
                        time.sleep(0.1)

                    elif sign['class_id'] == 1:
                        print("priority")
                        
                        outPs[0].send({'action': '1', 'speed': 0.06})
                        time.sleep(3.5)
                        outPs[0].send({'action': '1', 'speed': 0.09})
                        time.sleep(0.1)

                    elif sign['class_id'] == 2:
                        print("roundabout")
                        

                        # TODO: roundabout

                    elif sign['class_id'] == 3:
                        print("oneway")
                        

                    elif sign['class_id'] == 4:
                        print("highwaybegin")


                    elif sign['class_id'] == 5:
                        print("highwayend")
                        

                    elif sign['class_id'] == 6:
                        print("pedestrian crossing")

                        outPs[0].send({'action': '1', 'speed': 0.04})
                        time.sleep(0.5)
                        outPs[0].send({'action': '1', 'speed': 0.09})
                        time.sleep(0.1)

                    elif sign['class_id'] == 7:
                        print("park")

                        # TODO: call parking manouver

                    elif sign['class_id'] == 8:
                        print("do not enter")
                        outPs[0].send({'action': '1', 'speed': 0.0})
                        time.sleep(0.5)


                if deviation > 300:
                    self.current_steering_angle = 20.0
                    # outPs[0].send({'action': '2', 'steerAngle': 20.0} )
                elif deviation < -300:
                    self.current_steering_angle = -20.0
                    # outPs[0].send({'action': '2', 'steerAngle': -20.0} )
                else:
                    self.current_steering_angle = float(deviation/15)
                    # outPs[0].send({'action': '2', 'steerAngle': self.current_steering_angle} )

                print("Decision: finished processing data", str(count), " at ", str(time.ctime()))
                f = open("dec_log.txt", "a")
                f.write("Deviation: " + str(deviation) + "Angle: " + str(self.current_steering_angle))
                f.close()

                if self.current_state == "BASE":
                    outPs[0].send({'action': '2', 'steerAngle': self.current_steering_angle} )
                    time.sleep(0.25)
                    outPs[0].send({'action': '1', 'speed': 0.14} )
                    time.sleep(0.25)

                outPs[0].send({'action': '3', 'brake (steerAngle)': self.current_steering_angle} )

                outPs[1].send("I'm ready lane " + str(count))
                outPs[2].send("I'm ready object" + str(count))

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
