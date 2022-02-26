import cv2
import numpy as np
import sqlite3
import os

def createData():
    con = sqlite3.connect('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/data.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE student
               (id text, name text)''')
    con.commit()
    con.close()

def insertOrUpdateData(id1,name):
    con = sqlite3.connect('/Users/macbookpro/Workspace/Advance_python/week3/firstapplication/faceRecognition/data.db')
    cur = con.cursor()

    query = "SELECT id FROM student WHERE id='"+str(id1)+"'"
    cursorr = cur.execute(query)

    isRecordExist = 0

    for row in cursorr:
        isRecordExist = 1
        # print(row)
    # print(isRecordExist)
    
    if(isRecordExist == 0):
        query = "INSERT INTO student VALUES ('"+str(id1)+"','"+str(name)+"')"
    else:
        query = "UPDATE student SET name='"+str(name)+"' WHERE id='"+str(id1)+"'"
    # print(isRecordExist)
    cur.execute(query)
    con.commit()
    con.close()

# insertOrUpdateData('A37241','Viet')
# createData()
# load video
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

id = input("Enter your student code: ")
name = input("Enter your student name: ")
insertOrUpdateData(id,name)

count = 0;

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray,1.3,5)

    for(x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        count +=1
        cv2.imwrite('dataSet/'+str(id)+'_'+str(count)+'.jpg',gray[y:y+h,x:x+w])
        
    cv2.imshow('Data',frame)
    cv2.waitKey(1)

    if(count > 100):
        break

cap.release()
cv2.destroyAllWindows()
