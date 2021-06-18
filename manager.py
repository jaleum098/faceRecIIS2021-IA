import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import skimage as sk
import time

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model, load_model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.preprocessing.image import ImageDataGenerator

from imutils.object_detection import non_max_suppression

from facerecog_utils import *
from database_utils import *
from align import AlignDlib
from inception_blocks import *


# Load the pretrained model
def loadModel(filePath):
  pathCheck= os.path.exists(filePath)  
  FRModel = faceRecoModel(input_shape=(3,96,96))
  FRModel.load_weights(filePath)
  return FRModel

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

def generateFacesDataBase(dirpath, FRmodel, augmentations=4, output_name='facesData.npy'):
    encoded_database = {}
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            target_name = name.split('.')[0]
            file_path = str(root) + '/' + str(name)
            image = import_image(file_path)
            operations = augmentations+1
            for i in range(operations):
                this_name = target_name + '-' + str(i)
                if i>0:
                    image = apply_transform(image, num_transform=2)
                faces, face_pos, img_with_faces = get_faces_from_image(image)
                if not faces:
                    pass
                else:
                    face = faces[0]
                
                face_encoding = image_to_encoding(image, FRmodel)
                encoded_database[this_name] = face_encoding
                
    np.save(output_name, encoded_database)
    faceRecModel = encoded_database

def addNewFace(dirpath, FRModel, augmentations=4,output_name='facesData.npy'):
  encoded_images = {}
  for name in os.listdir(dirpath):
    print(name)
    target_name = name.split('.')[0]
    file_path = dirpath + str(name)
    image = import_image(file_path)
    operations = augmentations+1
    for i in range(operations):
        this_name = target_name + '-' + str(i)
        if i>0:
            image = apply_transform(image, num_transform=2)
        faces, face_pos, img_with_faces = get_faces_from_image(image)
        if not faces:
            pass
        else:
            face = faces[0]
        
        try:
          face_encoding = image_to_encoding(image, FRModel)
          encoded_images[this_name] = face_encoding
        except:
          print("No se encontraron Rostros")
  facesData.update(encoded_images)
  np.save(output_name, facesData)

def predictIndividual(face):
    time_start = time.time()
    plt.figure(figsize=(15,10))
    face_out = face_recognition(face, faceRecModel, facesData, plot=True, faces_out=True)
    time_end = time.time()
    print(face_out.keys())
    print('Total run time: %.2f ms' %((time_end-time_start)*1.e3))

faceRecModel = loadModel('Libraries/nn4.small2.v1.h5')
facesData = loadDataBase('facesData.npy')

#generateFacesDataBase('Database/', faceRecModel, augmentations=3, output_name='facesData.npy')

addNewFace('../Database/Diego/', faceRecModel, augmentations=3, output_name='facesData.npy')
addNewFace('../Database/Alejandro/', faceRecModel, augmentations=3, output_name='facesData.npy')
addNewFace('../Database/Michelle/', faceRecModel, augmentations=3, output_name='facesData.npy')
addNewFace('../Database/Stephanie/', faceRecModel, augmentations=3, output_name='facesData.npy')
print(facesData.keys())

'''time_start = time.time()
for file in os.listdir("../Datos de prueba/Diego/"):
    face = cv2.imread("../Datos de prueba/Diego/"+file, cv2.COLOR_RGB2BGR)
    predictIndividual(face) 
time_end = time.time()'''

