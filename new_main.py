from picamera2 import Picamera2
from cv2 import resize, INTER_LINEAR, imwrite, FONT_HERSHEY_SIMPLEX, LINE_AA, rectangle, putText
from src.perception.lane_detection.lane_detection import process_frame
from threading import Thread
from src.perception.object_classi.detect import load_labels, detect_objects
from tflite_runtime.interpreter import Interpreter
import serial
from multiprocessing import Pipe
from src.hardware.serialhandler.SerialHandlerProcess        import SerialHandlerProcess
import time
from src.utils.camerastreamer.CameraStreamerProcess         import CameraStreamerProcess
from LaneKeepingAlgorithm import lane_finding

def main():
    # -----------------------CONFIG-----------------------
    stream = True
    # ----------------------------------------------------
    print("Starting the car")
    camera = Picamera2()
    capture_config = camera.create_still_configuration()
    camera.configure(capture_config)
    camera.start()

    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()

    decSerialOut, decSerialIn   = Pipe(duplex = False) # decision making to serial
    shProc = SerialHandlerProcess([decSerialOut], [])     
    shProc.daemon = True
    shProc.start()

    current_speed = 0.0
    steering_angle = 0.0

    current_state = "BASE"

    if stream:
        visionStrOut, visionStrIn = Pipe(duplex=False)  # vision -> streamer
        streamProc = CameraStreamerProcess([visionStrOut], [])
        streamProc.daemon = True
        streamProc.start()

    try:
        while True:
            frame = camera.capture_array("main")
            resized_frame = resize(frame, dsize=(320, 320), interpolation=INTER_LINEAR)

            lane_finding_results = []
            lane_finding_thread = Thread(target=new_lane_detection, args=(resized_frame, lane_finding_results))

            object_detection_results = []
            object_detection_thread = Thread(target=object_detection, args=(resized_frame, interpreter,labels, object_detection_results))

            lane_finding_thread.start()
            object_detection_thread.start()
            
            lane_finding_thread.join()
            object_detection_thread.join()

            print("Deviation: ", lane_finding_results[0])
            print("Detected objects: ",object_detection_results[0])

            deviation = lane_finding_results[0] - 90
            res = object_detection_results[0]

            if stream:
                visionStrIn.send(["vigy", lane_finding_results[1]])

            handle_signs(res, decSerialIn)

            if deviation > 300:
                current_steering_angle = 20.0
            elif deviation < -300:
                current_steering_angle = -20.0
            else:
                current_steering_angle = float(deviation/15)

            if current_state == "BASE":
                    decSerialIn.send({'action': '2', 'steerAngle': current_steering_angle} )
                    time.sleep(0.25)
                    decSerialIn.send({'action': '1', 'speed': 0.12} )
                    time.sleep(0.25)

            decSerialIn.send({'action': '3', 'brake (steerAngle)': current_steering_angle} )

        


    except KeyboardInterrupt:
        if hasattr(shProc,'stop') and callable(getattr(shProc,'stop')):
            print("Process with stop",shProc)
            shProc.stop()
            shProc.join()
        else:
            print("Process witouth stop",shProc)
            shProc.terminate()
            shProc.join()

        if stream:
            if hasattr(streamProc,'stop') and callable(getattr(streamProc,'stop')):
                print("Process with stop",streamProc)
                streamProc.stop()
                streamProc.join()
            else:
                print("Process witouth stop",streamProc)
                streamProc.terminate()
                streamProc.join()

        print("vege")


def lane_detection(frame, output):
    deviation, final_frame = process_frame(frame=frame)
    output.append(deviation)
    output.append(final_frame)

def new_lane_detection(frame, output):
    angle, final_frame = lane_finding(frame)
    output.append(angle)
    output.append(final_frame)


def object_detection(frame,interpreter,labels, output):
    res = detect_objects(interpreter, frame, 0.8)
    for result in res:
        ymin, xmin, ymax, xmax = result['bounding_box']
        xmin = int(max(1,xmin * 320))
        xmax = int(min(320, xmax * 320))
        ymin = int(max(1, ymin * 320))
        ymax = int(min(320, ymax * 320))
        
        rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
        putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, 320-20)), FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,LINE_AA) 
    output.append(res)
    output.append(frame)


def handle_signs(res, pipe):
    for sign in res:
        print("Detected sign with id: ", sign['class_id'])
        if sign['class_id'] == 0:
            print("stopping")
            
            pipe.send({'action': '3', 'brake (steerAngle)': 0.0} )
            time.sleep(3)
            pipe.send({'action': '1', 'speed': 0.12} )
            time.sleep(0.2)
            pipe.send({'action': '1', 'speed': 0.09} )
            time.sleep(0.1)

        elif sign['class_id'] == 1:
            print("priority")
            
            pipe.send({'action': '1', 'speed': 0.06})
            time.sleep(0.5)
            pipe.send({'action': '1', 'speed': 0.09})
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

            pipe.send({'action': '1', 'speed': 0.04})
            time.sleep(0.5)
            pipe.send({'action': '1', 'speed': 0.09})
            time.sleep(0.1)

        elif sign['class_id'] == 7:
            print("park")

            # TODO: call parking manouver

        elif sign['class_id'] == 8:
            print("do not enter")
            pipe.send({'action': '1', 'speed': 0.0})
            time.sleep(0.5)


if __name__ == "__main__":
    main()