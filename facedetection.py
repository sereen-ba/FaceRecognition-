import cv2    #this a critical command without it nothing is going to work
import sys
import numpy as np

cascPath =sys.argv[0]

#Here we are declaaring the methodes and function that we will use
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# the image that you will be using must be saved in the same directory mostly in cv2.so
print('please make sure that the image that you will be using must be saved in the same directory mostly in cv2.so \n\n')

img = input('Enter you image file name as name.extantion ~ jpeg or png~ ')
# I used this image 'faces1.jpg' in my machine
pic1=cv2.imread(img)
gray = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)

scale_factor =1.3  

while 1:
    faces2 = faceCascade.detectMultiScale(gray,scale_factor,5)

    for (x, y, w, h) in faces2:                                    #her we are setting the face array to which will makes read to surround it
        cv2.rectangle(pic1, (x, y), (x+w, y+h), (255, 0, 0), 2)    #Drwing the rectangle and clearifing its dimenssion
        font = cv2.FONT_HERSHEY_SIMPLEX
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = pic1[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)                    #detecting the eyes array
        cv2.putText(pic1,'face',(x,y),font,2,(0,0,255),2,cv2.LINE_AA)    #writing a comment above the picture as wanted or killed etc..
        for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)      #drwaing a cicle around the eyes

         
    print('number of faces found{}'.format(len(faces2)))                 #stateing the number of people or faces were in the picture
    
    cv2.imshow('face',pic1)                                              #finally showing/displaying the image
  
    if cv2.waitKey()==2 & 0xff ==ord('q'):  ## delay for the specified milliseconds
        break
  
cv2.destroyAllWindows()                     ##This function will destroy all the previously created windows


##################    ################## ##################    ################## ##################    ################## ##################    ##################

#   Arthu: SEREEN BAHDAD
