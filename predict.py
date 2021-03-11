# import the necessary packages
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os


img_width, img_height = 224, 224
MODEL_PATH = os.path.sep.join(["output", "flower.model"])
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

print("[INFO] loading model...")
model = load_model(MODEL_PATH)

def predict(file):
  image = cv2.imread(file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224, 224))

  image = image.astype("float32")
  mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
  image -= mean

  preds = model.predict(np.expand_dims(image, axis=0))[0]
  i = np.argmax(preds)
  label = CLASSES[i]

  return label

daisy_t = 0
daisy_f = 0
rose_t = 0
rose_f = 0
sunflower_t = 0
sunflower_f = 0

for i, ret in enumerate(os.walk('./test-data')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print(filename)
    result = predict(ret[0] + '/' + filename)
    print(result)

