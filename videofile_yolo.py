# -*- coding: utf-8 -*-

#USE TESTVIDEOS git Repository for basic testing.

import cv2 #imports openCV library
from darkflow.net.build import TFNet #From the machine learning model, import the commands for detection
import numpy as np
import time #FPS/Time analysis

options = {
        
        'model':'cfg/yolo.cfg',
        'load':'bin/yolov2.weights',
        'threshold':0.3,
        'gpu': 2.0
        }

tfnet = TFNet(options) 

capture = cv2.VideoCapture("video.mp4") #Loads in video from the same directory
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

frame_width = int( capture.get(cv2.CAP_PROP_FRAME_WIDTH)) #Gather frame width and height
frame_height =int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID') #Formatting for displaying new video with detected objects

out = cv2.VideoWriter('outpy.avi',fourcc, 15, (frame_width,frame_height))
#writes video

while (capture.isOpened()):
    stime = time.time() #setting a running time fro FPS
    ret, frame = capture.read() # frame = current frame, and ret = next frame
    if ret: #If there is a next frame
        results = tfnet.return_predict(frame) #model detects objects, essentially a black box
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y']) #dictionary with the top left x/y coordinates
            br = (result['bottomright']['x'], result['bottomright']['y']) #same thing but with bottom right x/y coordinates
            label = result['label'] #labels (need to adjust for detecting humans only)
            frame = cv2.rectangle(frame, tl, br, color, 7) #OpenCV command for displaying all the bounding boxes
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) #Puts labels on each bounding box with confidence levels
        out.write(frame)
        #cv2.imshow('frame',frame)    
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'): #just waits for q as an interrupt
            break

capture.release() #essentially a close function for reading in the videos
out.release() #close function for writing the videos
cv2.destroyAllWindows() #clear every display when done
print("finished")
