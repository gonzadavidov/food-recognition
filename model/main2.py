from tensorflow.keras.models import Model
from keras.utils import normalize
from keras.utils import to_categorical
from keras.metrics import MeanIoU
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import random
from model import build_vgg19_unet

test_image_directory = 'test/images/'
test_mask_directory = 'test/masks/'

SIZE = 426
input_shape = (SIZE, SIZE, 3)
n_classes = 16

x_test = []
y_test = []

images = os.listdir(test_image_directory)
for i, image_name in enumerate(images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(test_image_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        x_test.append(np.array(image))

masks = os.listdir(test_mask_directory)
for i, mask_name in enumerate(masks):
    if(mask_name.split('.')[1] == 'jpg'):
        mask = cv2.imread(test_mask_directory+mask_name, 0)
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE, SIZE))
        y_test.append(np.array(mask))


x_test = np.expand_dims(x_test, axis=3)
x_test = normalize(x_test, axis=1)
y_test = np.expand_dims(np.array(y_test), axis=3) / 255.  #change the 255 if masks already normalized

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))  #check this line

#Load Model
model = build_vgg19_unet(input_shape, n_classes)
model.load_weights('food_recognition_50_epochs_test.hdf5')

#Calculate Accuracy
_, acc = model.evaluate(x_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

#IoU
y_pred = model.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

class_IoU = []
for i in range(n_classes):
  class_IoU[i] = values[i,i]/(np.sum(values[i,:]) + np.sum(values[:,i]))
  print("IoU for class %d is: %f", i, class_IoU[i])

#plt.imshow(train_images[0, :,:,0], cmap='gray')
#plt.imshow(train_masks[0], cmap='gray')