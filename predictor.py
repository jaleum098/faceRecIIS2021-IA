import cv2
import os
import numpy as np
import time
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from datetime import datetime

from keras import backend as K
K.set_image_data_format('channels_first')

from facerecog_utils import *
from database_utils import *
#from align import AlignDlib
from inception_blocks import *


faceout = {}
flag = True
cred = credentials.Certificate("./crendentials.json")
firebase_admin.initialize_app(cred,{"storageBucket": "face-recognition-cffc0.appspot.com"})

def assess():
    global flag
    global faceout

    for key,value in faceout.items():
        img = value[1]
        cv2.imwrite('upload.png',img)

        bucket = storage.bucket()
        blob = bucket.blob(f"{key}_{datetime.now()}")

        blob.upload_from_filename("./upload.png")
        blob.make_public()
        print(blob.public_url)

        db = firestore.client()
        if(key == "Not in database"):
            data = {
                u"descripcion":u"Se detectó un desconocido",
                u"estado":False,
                u"fecha": datetime.now(),
                u"imagen":blob.public_url,
                u"nombre":u"Desconocido"
            }
        else:
            data = {
                u"descripcion":u"Se detectó a {0}".format(key),
                u"estado":True,
                u"fecha": datetime.now(),
                u"imagen":blob.public_url,
                u"nombre":key
            }

        db.collection(u"historial").add(data)
    faceout = {}
    flag = True
    print('Registrados')
    return -1

def updateList(current_faceout):
    flag = True
    global faceout
    for key, value in current_faceout.items():
        for key2, value2 in faceout.items():
            if key == key2:
                value2[0]+=1
                value2[1] = value
                flag = False
                break
        if flag:
            faceout[key] = [1, value]
        flag = True
    print(faceout.keys())

def restart():
    global flag
    print('restart')
    threading.Timer(10, assess).start()
    flag = False
    return


def loadModel(filePath):
  FRModel = faceRecoModel(input_shape=(3,96,96))
  FRModel.load_weights(filePath)
  return FRModel

def predictIndividual(face):
    #time_start = time.time()
    #plt.figure(figsize=(15,10))
    face_out = face_recognition(face, faceRecModel, facesData, plot=True, faces_out=True)
    #time_end = time.time()
    #print(face_out.keys())
    return face_out
    #print('Total run time: %.2f ms' %((time_end-time_start)*1.e3))

def loadDataBase(filePath):
  pathCheck= os.path.exists(filePath)  
  if(pathCheck == False):
      print("Archivo no encontrado.\n")
      return {}
  print("Datos de rostros encontrados.")
  np.load.__defaults__=(None, True, True, 'ASCII')
  face_dict = np.load('facesData.npy').item()
  np.load.__defaults__=(None, False, True, 'ASCII')
  return face_dict

def initTensorFaces():
    img = cv2.imread('initFacesService.jpg',cv2.IMREAD_UNCHANGED)
    predictIndividual(img)
 
faceRecModel = loadModel('Libraries/nn4.small2.v1.h5')
facesData = loadDataBase('facesData.npy')
initTensorFaces()


while True:
    cap = cv2.VideoCapture('http://192.168.0.6:8080/video')
    ret, frame = cap.read()
    cap.release()

    frame = cv2.resize(frame, (720,360), interpolation = cv2.INTER_AREA)

    result = predictIndividual(frame)

    updateList(result)

    if flag:
      print(faceout)
      restart()

    time.sleep(0.1)


cv2.destroyAllWindows()