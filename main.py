# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#========================================================================
# SCRIPT USED FOR WIRING ALL COMPONENTS
#========================================================================
import sys

from src.behaviour.decision_making.decision_making import DecisionMakingProcess
from src.perception.object_classi.object_classi import ObjectDetectionProcess
from src.perception.object_classi.ComputerVisionProcess import ComputerVisionProcess
sys.path.append('.')

import time
import signal
from multiprocessing import Pipe, Process, Event 

# hardware imports
from src.hardware.camera.cameraprocess                      import CameraProcess
from src.hardware.camera.CameraSpooferProcess               import CameraSpooferProcess
from src.hardware.serialhandler.SerialHandlerProcess        import SerialHandlerProcess

# utility imports
from src.utils.camerastreamer.CameraStreamerProcess         import CameraStreamerProcess
from src.utils.remotecontrol.RemoteControlReceiverProcess   import RemoteControlReceiverProcess

# control imports
from src.actuation.test_control.ControlTest                 import ControlTest

# =============================== CONFIG =================================================
enableStream        =   False
enableCameraSpoof   =   False 
enableRc            =   True
enableDecMaking     =   False
enableControl       =   False

# =============================== INITIALIZING PROCESSES =================================
allProcesses = list()

if enableDecMaking:
    # =============================== HARDWARE ===============================================
    camLaneOut, camLaneIn = Pipe(duplex=False)  # camera -> vision/lane finding
    camObjectOut, camObjectIn = Pipe(duplex=False)  # camera -> vision/object detection

    laneCamOut, laneCamIn = Pipe(duplex=False) # vision/lane finding -> camera
    objectCamOut, objectCamIn = Pipe(duplex=False) # vision/object detection -> camera

    laneDecOut, laneDecIn = Pipe(duplex=False)  # vision/lane finding -> decision making
    objectDecOut, objectDecIn = Pipe(duplex=False)  # vision/object detection -> decision making

    declaneOut, decLaneIn = Pipe(duplex=False) # decision making -> vision/lane finding
    decObjectOut, decObjectIn = Pipe(duplex=False) # decision making -> vision/lane finding


    decSerialOut, decSerialIn   = Pipe(duplex = False) # decision making to serial


    # ======================EZT HAGYD MEG============

    shProc = SerialHandlerProcess([decSerialOut], [])     

    decProc = DecisionMakingProcess([laneDecOut, objectDecOut], [decSerialIn, decLaneIn, decObjectIn])
    
    if enableStream:
        visionStrOut, visionStrIn = Pipe(duplex=False)  # vision -> streamer
        
        streamProc = CameraStreamerProcess([visionStrOut], [])
        visionProcess = ComputerVisionProcess([camLaneOut, camObjectOut, declaneOut, decObjectOut],[laneCamIn,objectCamIn, laneDecIn, objectDecIn,visionStrIn])
        allProcesses.append(streamProc)
    else:
        visionProcess = ComputerVisionProcess([camLaneOut, camObjectOut, declaneOut, decObjectOut],[laneCamIn, objectCamIn, laneDecIn, objectDecIn])

    camProc = CameraProcess([laneCamOut, objectCamOut],[camLaneIn, camObjectIn])
    
    allProcesses.append(camProc)
    allProcesses.append(visionProcess)
    allProcesses.append(decProc)
    allProcesses.append(shProc)
    

else:
        
    # =============================== HARDWARE ===============================================
    if enableStream:
        camStR, camStS = Pipe(duplex = False)           # camera  ->  streamer

        if enableCameraSpoof:
            camSpoofer = CameraSpooferProcess([],[camStS],'vid')
            allProcesses.append(camSpoofer)

        else:
            camProc = CameraProcess([],[camStS])
            allProcesses.append(camProc)

        streamProc = CameraStreamerProcess([camStR], [])
        allProcesses.append(streamProc)
    # =============================== DATA ===================================================
    #LocSys client process
    # LocStR, LocStS = Pipe(duplex = False)           # LocSys  ->  brain
    # from data.localisationsystem.locsys import LocalisationSystemProcess
    # LocSysProc = LocalisationSystemProcess([], [LocStS])
    # allProcesses.append(LocSysProc)



    # =============================== CONTROL - RC =================================================
    if enableRc:
        rcShR, rcShS   = Pipe(duplex = False)           # rc      ->  serial handler

        # serial handler process
        shProc = SerialHandlerProcess([rcShR], [])
        allProcesses.append(shProc)

        rcProc = RemoteControlReceiverProcess([],[rcShS])
        allProcesses.append(rcProc)
        
    # =============================== CONTROL - PROBA =================================================
    if enableControl:
        inP, outP   = Pipe(duplex = False)          # controll-kliens   ->  serial handler
        
        shProc = SerialHandlerProcess([inP], [])    # controller - szerver
        allProcesses.append(shProc)
        
        ctrlProc = ControlTest([], [outP])          # controller - kliens
        allProcesses.append(ctrlProc)


# ===================================== START PROCESSES ==================================
print("Starting the processes!",allProcesses)
for proc in allProcesses:
    proc.daemon = True
    proc.start()


# ===================================== STAYING ALIVE ====================================
blocker = Event()  

try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        if hasattr(proc,'stop') and callable(getattr(proc,'stop')):
            print("Process with stop",proc)
            proc.stop()
            proc.join()
        else:
            print("Process witouth stop",proc)
            proc.terminate()
            proc.join()
