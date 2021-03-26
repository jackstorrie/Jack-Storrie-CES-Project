import argparse
import time
import cv2
import imutils
import numpy as np
import picamera
from gpiozero import Robot
from imutils.video import VideoStream

wheels = Robot(left=(7, 8), right=(9, 10))
camera = picamera.PiCamera()

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


def scan_image_for_objects(vs, net, CLASSES, COLORS, args, status):
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        coords = []
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
                coords[i] = cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        sorted_detections = []
        for i in (0, len(detections)):
            sorted_detections[i] = [detections[i], coords[i]]
         return sorted_detections, status


def decide_and_perform_next_movement(detections, stationary, non_detection_count):
    closest_object = detections[0]
    co_difference = 0
    for i in (0, len(detections)):
        current_co_difference = detections[i, 1].endX - detections[i, 1].startX
        if current_co_difference > co_difference:
            co_difference = current_co_difference
            closest_object = detections[i]
    if len(detections) == 0:
        if stationary:
            non_detection_count += 1
            if non_detection_count >= 10:
                status = 0
                wheels.stop()
        elif not stationary:
            non_detection_count += 1
            wheels.forward(0.8)
    elif stationary:
        non_detection_count += 1
        wheels.forward(0.8)
        stationary = False
    elif closest_object[1].startX < 100 & co_difference > 120:
        non_detection_count = 0
        wheels.right(0.8)
    elif closest_object[1].startX < 100 & co_difference < 120:
        non_detection_count = 0
        wheels.forward(0.8)
    elif closest_object[1].startX >= 100 & closest_object[1].startX <= 200 & co_difference > 120:
        non_detection_count = 0
        wheels.backward(0.8)
    elif closest_object[1].startX >= 100 & closest_object[1].startX <= 200 & co_difference < 120:
        non_detection_count = 0
        wheels.forward(0.8)
    elif closest_object[1].startX > 200 & co_difference > 120:
        non_detection_count = 0
        wheels.left(0.8)
    elif closest_object[1].startX < 200 & co_difference < 120:
        non_detection_count = 0
        wheels.forward(0.8)
    return stationary, non_detection_count, status


def main():
    status = 1
    stationary = True
    non_detection_count = 0
    vs = VideoStream(usePiCamera=True).start()
    CLASSES, COLORS, net, args = image_detection_setup()
    time.sleep(5.0)
    while status == 1:
        detections = scan_image_for_objects(vs, net, CLASSES, COLORS, args, status)
        decide_and_perform_next_movement(detections, stationary, non_detection_count, status)
    cv2.destroyAllWindows()
    vs.stop()
    return

