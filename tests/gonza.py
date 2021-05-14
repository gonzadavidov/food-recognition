import os
from pycocotools.coco import COCO
import skimage.io as io
import skimage.transform as transf
import time
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("./")

from dataset_filtering.filter_cats import filtered_cats

TRAIN_IMAGES_DIRECTORY = "../train/images"
TRAIN_ANNOTATIONS_PATH = "../train/annotations.json"

SIZE_X = 426
SIZE_Y = 426

# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Only obtain images annotated with water or pears
cat_names = ['apple', 'jam']
(cat_ids, cat_names, img_ids) = filtered_cats(coco, cat_names=cat_names)
n_cats = len(cat_names)

# Print filtered categories
n_imgs = len(img_ids)
print(f'{n_cats} categories, with {n_imgs} images')

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]

print(f'Starting to read images, about to do {n_imgs} iterations...')
x_train = np.zeros((SIZE_X, SIZE_Y, 3, n_imgs))
for k, rel_path in tqdm(enumerate(img_paths)):
    # Save images to the target directory of filtered data
    image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, rel_path)
    img = io.imread(image_path)
    img = transf.resize(img, (SIZE_X, SIZE_Y))
    x_train[:, :, :, k] = img

# Now for each image, obtain it's corresponding masks
# Create dictionary with cat_id as key and a number from 0 to n_channels - 1 as value
cat_dict = dict(zip(cat_ids, range(n_cats)))
for id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=id, catIds=cat_ids)
    annotations = coco.loadAnns(ann_ids)










from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import random
from model.model import build_vgg19_unet


SIZE = 426
input_shape = (SIZE, SIZE, 3)
n_classes = 16

x_train = np.expand_dims(x_train, axis=3)
x_train = normalize(x_train, axis=1)
y_train = np.expand_dims(np.array(y_train), axis=3) / 255.  #change the 255 if masks already normalized

print("Class values in dataset are:", np.unique(y_train))  #Check

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))  #Check this line

#Sanity check for images and masks
image_number = random.randint(0, len(x_train))
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.reshape(x_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')

#Create Model
model = build_vgg19_unet(input_shape, n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #Check this line
model.summary()

history = model.fit(x_train, y_train_cat, 
                    batch_size=16,
                    verbose=1,
                    epochs=50,
                    validation_data=(X_test, y_test_cat),
                    #class_weight=class_weights,
                    shuffle=False)

model.save('food_recognition_50_epochs_test.hdf5') #Saving a model with 50 epochs
