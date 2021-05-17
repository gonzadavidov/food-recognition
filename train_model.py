import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from dataset_filtering.data_generation import DataGeneration
from dataset_filtering.filter_cats import filtered_cats

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

print(tf.test.gpu_device_name())

SIZE_X = 128
SIZE_Y = 128
N_CATS = 16

INPUT_SHAPE = (SIZE_X, SIZE_Y, 3)
BATCH_SIZE = 16

TRAIN_IMAGES_PATH = "train/images"
TRAIN_ANNOTATIONS_PATH = "train/annotations.json"
TEST_IMAGES_PATH = "test/images"
TEST_ANNOTATIONS_PATH = "test/annotations.json"

def batch_generator(batchsize, images_path, annotation_path):
  i_img = 0
  coco = COCO(annotation_path)

  categories_ids, categories_names, img_ids = filtered_cats(coco, n=N_CATS)

  images = coco.loadImgs(img_ids)
  img_paths = [img["file_name"] for img in images]

  data_gen = DataGeneration(coco, SIZE_X, SIZE_Y, categories_ids)

  while i_img < len(img_paths):
    inputs = np.zeros((batchsize, SIZE_X, SIZE_Y, 3))
    outputs = np.zeros((batchsize, SIZE_X, SIZE_Y, N_CATS))

    for i in range(batchsize):
      inputs[i] = data_gen.x_sample(join(images_path, img_paths[i_img]))
      outputs[i] = data_gen.y_sample(img_ids[i_img])

      i_img += 1

    yield normalize(inputs), outputs

def conv_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dropout(0.1)(x)

  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_block(x, num_filters)
  return x

def build_vgg19_unet(input_shape, n_classes=1):
  """ Input """
  inputs = Input(input_shape)

  """ Pre-trained VGG19 Model """
  vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
  for layer in vgg19.layers:
    layer.trainable = False

  """ Encoder """
  s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
  s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
  s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
  s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

  """ Bridge """
  b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)

  """ Decoder """
  d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
  d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
  d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
  d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

  """ Output """
  outputs = Conv2D(n_classes, (1,1), padding="same", activation="sigmoid")(d4)

  model = Model(inputs, outputs, name="VGG19_U-Net")
  return model

model = build_vgg19_unet(INPUT_SHAPE, N_CATS)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MeanIoU(N_CATS)])
model.summary()

callbacks = [
             ModelCheckpoint("/content/logs/model_checkpoints/food_segmentation.h5", verbose=1, save_best_only=True),
             EarlyStopping(patience=3, monitor="val_loss"),
             TensorBoard(log_dir="/content/logs/tensorboard_logs")
]

history = model.fit(
                    batch_generator(BATCH_SIZE, TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS_PATH),
                    verbose=1,
                    epochs=50,
                    validation_data=batch_generator(BATCH_SIZE, TEST_IMAGES_PATH, TEST_ANNOTATIONS_PATH),
                    #class_weight=class_weights,
                    shuffle=True,
                    callbacks=callbacks)

model.save('food_recognition_50_epochs_test.hdf5') #Saving a model with 50 epochs