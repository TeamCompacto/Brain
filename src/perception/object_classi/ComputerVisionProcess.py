import socket
import struct
import time
import numpy as np



from threading import Thread

import cv2

from src.templates.workerprocess import WorkerProcess

# MAS IMPORTOK

from threading import Thread
import time
from pathlib import Path
import socket

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from src.perception.object_classi.models.experimental import attempt_load
from src.templates.workerprocess import WorkerProcess
from src.perception.object_classi.utils.datasets import LoadStreams, LoadImages
from src.perception.object_classi.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from src.perception.object_classi.utils.plots import plot_one_box
from src.perception.object_classi.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from src.perception.object_classi.utils.datasets import letterbox

class ComputerVisionProcess(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Process used for sending images over the network to a targeted IP via UDP protocol 
        (no feedback required). The image is compressed before sending it. 

        Used for visualizing your raspicam images on remote PC.
        
        Parameters
        ----------
        inPs : list(Pipe) 
            List of input pipes, only the first pipe is used to transfer the captured frames. 
        outPs : list(Pipe) 
            List of output pipes (not used at the moment)
        """
        super(ComputerVisionProcess,self).__init__( inPs, outPs)
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(ComputerVisionProcess,self).run()

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the sending thread.
        """
        self.img_size = 320
        streamTh = Thread(name='ObjectDetectionThread',target = self._object_detection_thread, args= (self.inPs[0], ))
        streamTh.daemon = True
        self.threads.append(streamTh)

        
    # ===================================== SEND THREAD ==================================
    def _object_detection_thread(self, inP):
        """Sending the frames received thought the input pipe to remote client by using the created socket connection. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe to read the frames from CameraProcess or CameraSpooferProcess. 
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        

        while True:
            stamp, image = inP.recv()

            print(type(image))
            print(image.shape)
            print(image.dtype)

            image = image.astype(np.float32)

            print(type(image))
            print(image.shape)
            print(image.dtype)

            img0 = image

            img = letterbox(img0, new_shape = (self.img_size, self.img_size))[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # INNEN IROM BE A DOLGOKAT

            weights = "yolov7-tiny.pt"
            img_size = 320
            conf = 0.3
            device = 'cpu'

            set_logging()
            device = select_device(device)

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = 320 #check_img_size(imgsz, s=stride)  # check img_size

            print("Model loaded")

            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Csak az image - val kell dolgozzak

            img = torch.from_numpy(img).to(device)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)


            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=None)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred)#, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, img0)

            print("eddig")

            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', img0

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    save_conf = True

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        #ezt kell elkuldjem
                        string_to_send = ""
                        string_to_send += ('%g ' * len(line)).rstrip() % line + '\n'
                        print(string_to_send)

                        # Stream results
                        view_img = True
                        if view_img:
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)


            
