import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/faceRecognition/recognizer/trainingData.yml')

def getProfile(id):
    con = sqlite3.connect('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/faceRecognition/data.db')
    cur = con.cursor()

    query = "SELECT * FROM student WHERE id='"+str(id)+"'"
    cursorr = cur.execute(query)

    profile = None

    for row in cursorr:
        profile = row
    
    con.close()
    return profile

cap = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]

        id, confidence = recognizer.predict(roi_gray)
        id = 'A' + str(id)
        # print('id',id)
        if confidence< 40:
            profile = getProfile(id)
            print(profile)
            if(profile != None):
                cv2.putText(frame,""+str(profile[0])+"_"+str(profile[1]),(x+10,y+h+30),fontface,1,(0,255,0),2)
            
        else:
            cv2.putText(frame,"Unknown",(x+10,y+h+30),fontface,1,(0,0,255),2)
    cv2.imshow('SHOW',frame)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()