from operator import index
import numpy as np
from flask import Flask, request,jsonify
from flask_cors import cross_origin,CORS
import pickle
import os
import pandas as pd
import dlib
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import io
import base64 
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.utils import to_categorical
from numpy import load
import base64
from imageio import imread
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib
import base64
from numpy import load

frontalFaceDetector = dlib.get_frontal_face_detector()
def extract_image(image,size=(160,160)):
  image = image.convert("RGB")
  img = np.asarray(image)
  allFaces = frontalFaceDetector(img,1)
  if(len(allFaces)==0):
    return np.array([])
  for curFace in allFaces:
    x1,y1,x2,y2 = int(curFace.left()),int(curFace.top()),int(curFace.right()),int(curFace.bottom())
    face=img[y1:y2,x1:x2]
    face = Image.fromarray(face)
    face = face.resize(size)
    face = np.array(face)
    return face

def convertImage(base64_data):
   encoded_data = base64_data.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = Image.fromarray(img)
   return img

def get_embedding(model,face_pixels):
  face_pixels = face_pixels.astype('float32')
  mean,std = face_pixels.mean(),face_pixels.std()
  face_pixels = (face_pixels - mean) / std
  samples = np.expand_dims(face_pixels,axis=0)
  yhat = model.predict(samples)
  return yhat[0]


def save_face_data(images):
  print("saving faces")
  data = load("./faces-dataset.npz")
  X_train,y_train = data['arr_0'],data['arr_1']
  X_train_list,y_train_list = [],[]
  y_val = y_train[-1] + 1
  for image in images:
    image = convertImage(image)
    image = extract_image(image)
    X_train_list.append(list(image))
    y_train_list.append(y_val)
  np.savez_compressed('./faces-dataset.npz',np.array(X_train_list), np.array(y_train_list))



emmodel = load_model("./facenet_keras.h5")
def save_face_embeddings_data():
  print("saving embeddings")
  data = load("./faces-dataset.npz")
  X_train,y_train = data['arr_0'],data['arr_1']
  # X_train,y_train = a,b
  X_train_embeddings = []
  y_train_embeddings = []
  for face_pixels in X_train:
    # face_pixels = np.expand_dims(face_pixels,axis=0)
    embedding = get_embedding(emmodel,face_pixels)
    X_train_embeddings.append(embedding)
  for y in y_train:
    y_train_embeddings.append(y)
  # return X_train_embeddings
  data = load("./faces-embeddings-dataset.npz")

  X_train,y_train = data['arr_0'],data['arr_1']
  X_em_total = []
  y_em_total = []
  for myx in X_train:
    X_em_total.append(myx)
  for myy in y_train:
    y_em_total.append(myy)
  for myx in X_train_embeddings:
    X_em_total.append(myx)
  for myy in y_train_embeddings:
    y_em_total.append(myy)
  np.savez_compressed('./faces-embeddings-dataset.npz',np.array(X_em_total), np.array(y_em_total))






def train_model():
  print("training models")
  data = load("./faces-embeddings-dataset.npz")
  X_train,y_train = data['arr_0'],data['arr_1']
  in_encoder = Normalizer(norm='l2')
  X_train_tnf = in_encoder.transform(X_train)
  svc = SVC(kernel="linear",probability=True)
  svc.fit(X_train_tnf,y_train)
  joblib.dump(svc, './svc.pkl')
  print("done") 



def retrain_model(images):
  save_face_data(images)
  save_face_embeddings_data()
  train_model()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/")
def home():
    return "<h2>Hellop</h2>"


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predict():
    if request.method == "GET":
        thisdict = jsonify({"brand": "Ford","model": "Mustang","year": 1964})
        return thisdict

    if request.method == "POST":
        # file = open(,'r')
        model = joblib.load('./svc.pkl')
        my_dict  = request.get_json(force=True)
        print(my_dict["image"],my_dict)
        im = convertImage(my_dict["image"])
        im = extract_image(im)
        if(len(im)==0):
          return jsonify({"prediction":int(-1)})
        
        # im = np.expand_dims(im,axis=0)
        im = get_embedding(emmodel,im)
        
        im = np.expand_dims(im,axis=0)
        ans = model.predict(im)
        print("Ans ",ans)
        return jsonify({"prediction":int(ans[0])})


@app.route("/train", methods=["POST", "GET"])
@cross_origin()
def train():
    if request.method == "GET":
        return "<h1>TRAIN</h1>"

    if request.method == "POST":
      my_dict  = request.get_json(force=True)
      if(len(my_dict["images"]) <= 5):
        return jsonify({"error": "very few images to train the model please add more images"})
      retrain_model(my_dict["images"])
      return "successfully trained"

app.run(debug=True)