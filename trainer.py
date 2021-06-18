import os
import numpy as np

from keras import backend as K
K.set_image_data_format('channels_first')

from facerecog_utils import *
from database_utils import *
from align import AlignDlib
from inception_blocks import *


def loadModel(filePath):
  #pathCheck= os.path.exists(filePath)  
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

def addNewFace(dirpath,FRModel,augmentations=3,output_name='facesData.npy'):
  facesData = loadDataBase('facesData.npy')
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
          continue
  facesData.update(encoded_images)   
  np.save(output_name, facesData)

#addNewFace('Datos de prueba/Diego/', faceRecModel, augmentations=3, output_name='facesData.npy')