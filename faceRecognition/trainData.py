import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataSet'

def getImageWithId(path):
    imgPaths = [os.path.join(path, f) for f in  os.listdir(path)]
    # print(imgPaths)
    faces = []
    ids = []
    for imgPathItem in imgPaths:
        faceImg = Image.open(imgPathItem).convert('L')
        faceNp = np.array(faceImg,'uint8')

        idItem = int(imgPathItem.split('/')[1].split('_')[0][1:])

        faces.append(faceNp)
        ids.append(idItem)
    return faces,ids

faces, ids = getImageWithId(path)
recognizer.train(faces, np.array(ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()