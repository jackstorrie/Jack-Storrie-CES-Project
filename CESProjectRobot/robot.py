from gpiozero import Robot
import picamera
from time import sleep
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

wheels = Robot(left=(7, 8), right=(9, 10))
camera = picamera.PiCamera()


def decide_next_movement(stream):
    return


def image_detection_setup():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    return CLASSES, COLORS, net, args


def scan_image_for_objects(vs, net, CLASSES, COLORS, args):
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return vs, net, CLASSES, COLORS, args, detections


def deal_with_detections(param):
    pass


def main():
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()
    #   image_detection_setup()
    #   deal_with_detections((scan_image_for_objects(image_detection_setup()[5])))
    movement_test_square()
    movement_test_circle()
    cv2.destroyAllWindows()
    vs.stop()
    return


def movement_test_square():
    wheels.forward(2)
    sleep(2)
    wheels.stop()
    for i in range(0, 2):
        wheels.right(2)
        sleep(2)
        wheels.stop()
    return


def movement_test_circle():
    wheels.forward(2)
    sleep(2)
    wheels.right(2)
    sleep(6)
    wheels.stop()
    return
