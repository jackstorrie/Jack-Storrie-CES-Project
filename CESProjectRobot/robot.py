from gpiozero import Robot
import picamera
from time import sleep

front_wheels = Robot(left=(7, 8), right=(9, 10))
back_wheels = Robot(left=(11, 12), right=(13, 14))
camera = picamera.PiCamera()


def main():
    camera.start_preview()
    stream = picamera.PiCameraCircularIO(camera, seconds=1)
    camera.start_recording(stream, format='h264')
    # detection and movement
    movement_test_square()
    movement_test_circle()
    camera.stop_recording()
    camera.stop_preview()
    return


def movement_test_square():
    front_wheels.forward(2)
    back_wheels.forward(2)
    sleep(2)
    front_wheels.stop()
    back_wheels.stop()
    for i in range(0, 2):
        front_wheels.right(2)
        back_wheels.right(2)
        sleep(2)
        front_wheels.stop()
        back_wheels.stop()
    return


def movement_test_circle():
    front_wheels.forward(2)
    back_wheels.forward(2)
    sleep(2)
    front_wheels.right(2)
    back_wheels.forward(2)
    sleep(6)
    front_wheels.stop()
    back_wheels.stop()
    return
