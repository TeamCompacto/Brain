# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import re
import cv2
import numpy as np

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  boxes = interpreter.get_output_details()[0] 
  classes = interpreter.get_output_details()[0] 
  scores = interpreter.get_output_details()[0] 
  count = interpreter.get_output_details()[0] 
  print(len(boxes))
  print(len(classes))
  print(len(scores))
  print(len(count))

  print(boxes)
  print(classes)
  print(scores)
  print(count)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
  # labels = load_labels()
  # interpreter = Interpreter('detect.tflite')
  # interpreter.allocate_tensors()
  # _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # img = cv2.imread('kep.jpg')
  # res = detect_objects(interpreter, img, 0.8)
  # print(res)

  # for result in res:
  #     ymin, xmin, ymax, xmax = result['bounding_box']
  #     xmin = int(max(1,xmin * CAMERA_WIDTH))
  #     xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
  #     ymin = int(max(1, ymin * CAMERA_HEIGHT))
  #     ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

  #     cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
  #     cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)

  # cv2.imsave(frame, 'kep_new.jpg')
  base_options = core.BaseOptions(file_name='detect.tflite')
  detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.5)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)


  image = cv2.imread('kep.jpg')

  image = cv2.flip(image, 1)

  # Convert the image from BGR to RGB as required by the TFLite model.
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Create a TensorImage object from the RGB image.
  input_tensor = vision.TensorImage.create_from_array(rgb_image)

  # Run object detection estimation using the model.
  detection_result = detector.detect(input_tensor)

  print(detection_result)


if __name__ == "__main__":
    main()