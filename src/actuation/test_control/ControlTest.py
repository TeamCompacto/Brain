import json
import time
from threading import Thread

from src.templates.workerprocess import WorkerProcess

class ControlTest(WorkerProcess):
    def __init__(self, inPs, outPs):
        super(ControlTest,self).__init__(inPs, outPs)

        # alapbol probakent menjen elore -- 2 -es action azt csinalja
        self.data = {}
        self.set_data(
            action = '2', 
            type = 'speed', 
            param = float(30)
            )
        
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
    
    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the read thread to transmite the received messages to other processes. 
        """
        sendTh = Thread(name='ControlTestThread',target = self._send_command, args = (self.outPs, ))
        self.threads.append(sendTh)
        
    def _send_command(self, outPs): 
        try:
            # # amig lehet, kuldje el a jelenlegi statuszt
            # while True: 
            #     command =  json.loads(self.data)

            #     for outP in outPs:
            #         outP.send(command)
                   
            #     # masodpercenkent kuld
            #     time.sleep(1)
            outPs[0].send({'action': '1', 'speed': 0.12}  )
            time.sleep(2)
            outPs[0].send({'action': '1', 'speed': 0}  )
        except Exception as e:
            print("Baj van teso (ControlTest - pipeon valo kuldesnel): " +  str(e))
            
    
    