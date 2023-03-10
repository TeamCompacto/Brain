import json
import time
from threading import Thread

from src.templates.workerprocess import WorkerProcess

class ControlTest(WorkerProcess):
    def __init__(self, inPs, outPs):
        super(ControlTest,self).__init__(inPs, outPs)

        # alapbol probakent menjen elore -- 2 -es action azt csinalja
        self.data = {}
        
        self.speed = float(0)
        self.angle = float(0)
        
        # init base position 
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )

        
    def set_data(self, action, type, param):
        """Beállítja a data-t:  
                - action: melyik státuszt hajtsa végre ('1', '2', ...) -- string 
                - type: az actionhoz milyen param. tartozik (steerAngle, speed) -- string 
                - param: a paraméterezés értéke (pl 30.0, 40.0, ...) -- float 
        """
        self.data['action'] = str(action)
        self.data[str(type)] = float(param)
        
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(ControlTest,self).run()
    
    # ----------------------- BASIC CONTROL COMMANDS -----------------------

    def send_speed(self):
        self.outPs[0].send({'action': '1', 'speed': self.speed}  )
        
    def send_angle(self):
        self.outPs[0].send({'action': '2', 'steerAngle': self.angle}  )
    
    def send_break(self):
        self.outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )

            
    # ------------------------------ CONTROL  ------------------------------
            
    def update_controls(self, new_speed, new_steering_angle):
        self.speed = new_speed
        self.angle = new_steering_angle
        
        if self.speed == 0:
            self.send_break()
        else :
            self.send_speed()
            self.send_angle()
    
    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the read thread to transmite the received messages to other processes. 
        """
        sendTh = Thread(name='ControlTestThread',target = self._send_command, args = (self.outPs, ))
        self.threads.append(sendTh)
        
    def _send_command(self, outPs): 
        try:
            # turn_90_degrees(outPs, direction='right')
            self.update_controls(0.15, 0.0)
            time.sleep(2)
            self.update_controls(0.0, 0.0)
            # time.sleep(1)
            # turn_90_degrees(outPs, direction='right')
            # time.sleep(1)
            # turn_90_degrees(outPs, direction='right')
            # time.sleep(1)
            # turn_90_degrees(outPs, direction='right')
            # time.sleep(1)

            # # amig lehet, kuldje el a jelenlegi statuszt
            # while True: 
            #     command =  json.loads(self.data)

            #     for outP in outPs:
            #         outP.send(command)
                   
            #     # masodpercenkent kuld
            #     time.sleep(1)
            
        except Exception as e:
            # outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )
            self.update_controls(0.0, 0.0)
            print("Baj van teso (ControlTest - pipeon valo kuldesnel): " +  str(e))
        finally:
             outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )
             
        


def forvard(outPs, speed=0.1, duration=2):
        outPs[0].send({'action': '1', 'speed': speed}  )
        time.sleep(duration)
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )

def turn(outPs, speed=0.1, duration=2, angle=10.0):
        print("Turning")
        outPs[0].send({'action': '1', 'speed': speed}  )
        outPs[0].send({'action': '2', 'steerAngle': angle}  )
        time.sleep(duration)
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )

def turn_90_degrees(outPs, direction):
    if direction == 'right':
        outPs[0].send({'action': '1', 'speed': 0.1}  )
        outPs[0].send({'action': '2', 'steerAngle': 20.0}  )
        time.sleep(3.26)
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )
    if direction == 'left':
        outPs[0].send({'action': '1', 'speed': 0.1}  )
        outPs[0].send({'action': '2', 'steerAngle': 20.0}  )
        time.sleep(3.26)
        outPs[0].send({'action': '3', 'brake (steerAngle)': 0.0}  )
            