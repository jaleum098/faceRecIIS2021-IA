import cv2
import os
import numpy as np
import urllib
from trainer import addNewFace

from keras import backend as K
K.set_image_data_format('channels_first')

from facerecog_utils import *
from database_utils import *

from inception_blocks import *

from flask import Flask, request, jsonify 
from flask_cors import CORS


def loadModel(filePath):  
  FRModel = faceRecoModel(input_shape=(3,96,96))
  FRModel.load_weights(filePath)
  return FRModel

def predictIndividual(face):
    #time_start = time.time()
    #plt.figure(figsize=(15,10))
    face_out = face_recognition(face, faceRecModel, facesData, plot=True, faces_out=True)
    #time_end = time.time()
    print(face_out.keys())
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

app = Flask(__name__)
CORS(app)

@app.before_first_request
def loadAI():
    global faceRecModel
    global facesData
    faceRecModel = loadModel('Libraries/nn4.small2.v1.h5')
    facesData = loadDataBase('facesData.npy')
    initTensorFaces()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', "Origin, X-Requested-With, Content-Type, Accept, Authorization, Access-Control-Allow-Credentials")
    response.headers.add('Access-Control-Allow-Methods', "GET, POST, PUT, DELETE")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/addFace', methods = ['GET', 'POST'])
def addFace():

    data = request.json
    name = data["name"]
    path = '../Database/'+name+'/'

    os.makedirs(path,exist_ok=True)

    index = 0
    for url in data["list"]:
        url_response = urllib.request.urlopen(url)
        img = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)
        cv2.imwrite(f"{path}{name}_{index}.jpg",img)
        index+=1
    
    addNewFace(path,faceRecModel,3,'facesData.npy')
    response = jsonify({'Status':True})
    return  response
 



@app.route('/test', methods = ['GET', 'POST'])
def getMask():  
    response = jsonify({'Status':True})
    return  response
   
if __name__ == '__main__':
    
    app.run(debug=True, port=7000) #run app in debug mode on port 5000s