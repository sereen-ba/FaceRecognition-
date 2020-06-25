import numpy as np
import cv2
import os

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
       # Draw a rectangle around the eyes  
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)      #drwaing a cicle around the eyes

         
    # Display the resulting frame
    cv2.imshow('Video', frames)
    
    if cv2.waitKey(1)==27:                   ## delay for the specified milliseconds that means we pessed the esc key 
        break
video_capture.release()
  
cv2.destroyAllWindows()                     ##This function will destroy all the previously created windows


##################    ################## ##################    ################## ##################    ################## ##################    ##################

#   Arthu: SEREEN BAHDAD
