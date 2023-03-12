from picamera2 import Picamera2
from cv2 import resize, INTER_LINEAR, imwrite, FONT_HERSHEY_SIMPLEX, LINE_AA, rectangle, putText
from src.perception.object_classi.lane_finding import process_frame
from threading import Thread
from src.perception.object_classi.detect import load_labels, detect_objects
from tflite_runtime.interpreter import Interpreter
import serial
from multiprocessing import Pipe
from src.hardware.serialhandler.SerialHandlerProcess        import SerialHandlerProcess
import time
from src.utils.camerastreamer.CameraStreamerProcess         import CameraStreamerProcess

CURRENT_STATE = "BASE"


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
            lane_finding_thread = Thread(target=lane_detection, args=(resized_frame, lane_finding_results, ))

            object_detection_results = []
            object_detection_thread = Thread(target=object_detection, args=(resized_frame, interpreter,labels, object_detection_results, ))

            lane_finding_thread.start()
            object_detection_thread.start()
            
            lane_finding_thread.join()
            object_detection_thread.join()

            

            deviation = lane_finding_results[0]
            res = object_detection_results[0]

            print("Deviation: ", deviation)
            print("Detected objects: ",res)

            if stream:
                visionStrIn.send(["vigy", lane_finding_results[1]])

            handle_signs(res, decSerialIn)

            if deviation > 500:
                current_steering_angle = 15.0
            elif deviation < -500:
                current_steering_angle = -15.0
            elif deviation > 300:
                current_steering_angle = 15.0
            elif deviation < -300:
                current_steering_angle = -15.0
            elif deviation > 100:
                current_steering_angle = 10.0
            elif deviation < -100:
                current_steering_angle = -10.0
            elif deviation > 50:
                current_steering_angle = 5.0
            elif deviation < -50:
                current_steering_angle = -5.0
            else:
                current_steering_angle = float(0)

            if CURRENT_STATE == "BASE":
                    decSerialIn.send({'action': '2', 'steerAngle': current_steering_angle} )
                    time.sleep(0.25)
                    decSerialIn.send({'action': '1', 'speed': 0.16} )
                    time.sleep(0.25)

            decSerialIn.send({'action': '3', 'brake (steerAngle)': current_steering_angle} )
            time.sleep(0.1)

    except KeyboardInterrupt:
        decSerialIn.send({'action': '3', 'brake (steerAngle)': 0.0} )

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

    if CURRENT_STATE == "BASE":
        deviation, final_frame = process_frame(frame=frame)
        output.append(deviation)
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
            CURRENT_STATE = "INTERSECTION"
            print(CURRENT_STATE)
            intersection_go_forward(pipe)
            CURRENT_STATE = "BASE"
            print(CURRENT_STATE)

        elif sign['class_id'] == 2:
            print("roundabout")

        elif sign['class_id'] == 3:
            print("oneway")
            

        elif sign['class_id'] == 4:
            print("highwaybegin")


        elif sign['class_id'] == 5:
            print("highwayend")
            

        elif sign['class_id'] == 6:
            print("pedestrian crossing")

            pipe.send({'action': '1', 'speed': 0.04})
            time.sleep(1)
            pipe.send({'action': '1', 'speed': 0.09})
            time.sleep(0.1)

        elif sign['class_id'] == 7:
            print("park")

            # TODO: call parking manouver

        elif sign['class_id'] == 8:
            print("do not enter")
            pipe.send({'action': '1', 'speed': 0.0})
            time.sleep(0.5)

def intersection_go_left(pipe):
    pass


def intersection_go_forward(pipe):
    pipe.send({'action': '2', 'steerAngle': 0.0})
    pipe.send({'action': '1', 'speed': 0.10})
    time.sleep(2)
    pipe.send({'action': '1', 'speed': 0.09})
    time.sleep(0.1)



if __name__ == "__main__":
    main()