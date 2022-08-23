from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from gtts import gTTS

import pygame

import RPi.GPIO as GPIO

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BOARD)

servo_pin = 11 # connect servo pin to 11th pin of raspberry
red_led = 15 # connect RED LED to 13th pin of raspberry
green_led = 13 # connect GREEN LED to 15th pin of raspberry

GPIO.setup(servo_pin,GPIO.OUT) 
GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(green_led, GPIO.OUT)



servo = GPIO.PWM(servo_pin,50) 

servo.start(7.5)

# play the welcome audio file

pygame.mixer.init()
pygame.mixer.music.load("audio/welcome-audio.mp3")
pygame.mixer.music.play()


# detect mask presence
def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


print("please wait for 20 seconds, the mask detector model file is loading ...")


prototxtPath = "/home/pi/Desktop/Mask-Detection/face_detector/deploy.prototxt"
weightsPath = "/home/pi/Desktop/Mask-Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

####################################################################

for i in range(0,10):
    frame = vs.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(0.1)
    
########################################################################
while True:

    frame = vs.read()
    frame_copy = frame.copy()
    #frame = imutils.resize(frame, width=400)

   

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        if(mask < withoutMask):

            label = "No Mask"
            color = (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame_copy, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            print('NO MASK')


            servo.ChangeDutyCycle(7.5)
            GPIO.output(red_led,True)
            GPIO.output(green_led,False)
            
            pygame.mixer.music.load("audio/no-mask.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue

        else:

            label = "Mask" 
            color = (0, 255, 0)
            color1 = (0, 0, 255)
            color2 = (255, 0, 0)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame_copy, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            
            print('MASK')

            pygame.mixer.music.load("audio/entry.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue

            label = "Mask" 
                
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame_copy, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            GPIO.output(red_led,False)
            GPIO.output(green_led,True)
            servo.ChangeDutyCycle(2.5)
            print("Gate Is Opening...")

            label2 = "Gate Is Opening"
                    
            cv2.putText(frame_copy, label2, (startX, startY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color2, 2)
            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            pygame.mixer.music.load("audio/gate-opening.mp3")
            pygame.mixer.music.play()
                    
            time.sleep(5)
            servo.ChangeDutyCycle(7.5)
            GPIO.output(red_led,True)
            GPIO.output(green_led,False)
                    
            label2 = "Gate Is Opening / Closing"
            cv2.putText(frame_copy, label2, (startX, startY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color2, 2)
            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            pygame.mixer.music.load("audio/gate-closing.mp3")
            pygame.mixer.music.play()
                    
            print("Gate Is Closing...")
            time.sleep(5)


        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame_copy, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

            
