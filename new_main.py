from picamera2 import Picamera2
from cv2 import resize, INTER_LINEAR, imwrite
from src.perception.object_classi.lane_finding import process_frame
from threading import Thread
from src.perception.object_classi.detect import load_labels, detect_objects
from tflite_runtime.interpreter import Interpreter



def main():
    # -----------------------CONFIG-----------------------
    stream = False
    print("Starting the car")
    camera = Picamera2()
    capture_config = camera.create_still_configuration()
    camera.configure(capture_config)
    camera.start()

    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()

    try:
        while True:
            frame = camera.capture_array("main")
            resized_frame = resize(frame, dsize=(320, 320), interpolation=INTER_LINEAR)

            lane_finding_results = []
            lane_finding_thread = Thread(target=lane_detection, args=(resized_frame, lane_finding_results))

            object_detection_results = []
            object_detection_thread = Thread(target=object_detection, args=(resized_frame, interpreter,labels, object_detection_results))

            lane_finding_thread.start()
            object_detection_thread.start()
            
            lane_finding_thread.join()
            object_detection_thread.join()

            print(lane_finding_results)
            print(object_detection_results)


    except KeyboardInterrupt:
        print("vege")


def lane_detection(frame, output):
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
        
        cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
        cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, 320-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 
    output.append(res)
    output.append(frame)

if __name__ == "__main__":
    main()