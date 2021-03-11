import os
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, InceptionResNetV2, Xception
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("Agg")

BASE_PATH = "dataset"
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
BATCH_SIZE = 16
BASE_CSV_PATH = "output"

MODEL_PATH = os.path.sep.join(["output", "flower.model"])
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])

def plot_training(H, N, plotPath):
  # construct a plot that plots and saves the training history
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  plt.savefig(plotPath)

trainPath = os.path.sep.join([BASE_PATH, TRAIN])
valPath = os.path.sep.join([BASE_PATH, VAL])
testPath = os.path.sep.join([BASE_PATH, TEST])

totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

trainAug = ImageDataGenerator(
  rotation_range=30,
  zoom_range=0.15,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.15,
  horizontal_flip=True,
  fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

trainGen = trainAug.flow_from_directory(
  trainPath,
  class_mode="categorical",
  target_size=(224, 224),
  color_mode="rgb",
  shuffle=True,
  batch_size=BATCH_SIZE)

valGen = valAug.flow_from_directory(
  valPath,
  class_mode="categorical",
  target_size=(224, 224),
  color_mode="rgb",
  shuffle=False,
  batch_size=BATCH_SIZE)

testGen = valAug.flow_from_directory(
  testPath,
  class_mode="categorical",
  target_size=(224, 224),
  color_mode="rgb",
  shuffle=False,
  batch_size=BATCH_SIZE)

baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

baseModel.trainable = True
for layer in baseModel.layers[-15:]:
  layer.trainable = True

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,
  metrics=["accuracy"])

H = model.fit(
  x=trainGen,
  steps_per_epoch=totalTrain // BATCH_SIZE,
  validation_data=valGen,
  validation_steps=totalVal // BATCH_SIZE,
  epochs=50)

print("[INFO] evaluating after fine-tuning network...")
predIdxs = model.predict(x=testGen, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))
# plot_training(H, 50, UNFROZEN_PLOT_PATH)

# serialize the model to disk
print("[INFO] serializing network...")
model.save(MODEL_PATH, save_format="h5")