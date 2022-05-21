import numpy as np
import  os
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import time
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
import pickle

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

debugg=False

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    
    
    
    return normalized_features
if debugg:
        #we see thw lenght that the model generates taking a photo we have in a file in the computer
        features = extract_features('sample_images/cat.jpg', model)
        print(len(features))
        features2 = extract_features('sample_images/cat2.jpg', model)
        print(len(features2))
        ##IN TUTORIAL SAYS THAT SHOULD BE 2048!!!!

#get image files from directory
##CHANGE NAMES AND STRUCTUREEEE
extensiones=[ '.JPG' , '.PNG','.jpg','.JPEG', '.png', '.jpeg']
def get_images(directory_direction):
    img_list=[]
    contador=1
    for root, directories,filenames in os.walk(directory_direction):
        for filename in filenames:
            if any(ex in filename for ex in extensiones):
                img_list.append(os.path.join(root,filename))
                ++contador
    return img_list

# path to the datasets
root_dir = 'Caltech101'  
filenames = sorted(get_images(root_dir))
list_of_features = []
for i in tqdm_notebook(range(len(filenames))):
    list_of_features.append(extract_features(filenames[i], model))

file_name='features-caltech101.pkl'
f = open(file_name,'wb')
pickle.dump(list_of_features,f)
f.close()

