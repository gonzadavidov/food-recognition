# Test of DataGeneration class

import os
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import matplotlib.pyplot as plt
import cv2

from dataset_filtering.data_generation import DataGeneration
from dataset_filtering.filter_cats import filtered_cats


TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"
VAL_IMAGES_DIRECTORY = "data/val/images"
VAL_ANNOTATIONS_PATH = "data/val/annotations.json"

SIZE_X = 426
SIZE_Y = 426

# Build coco objects pointing to the annotations
coco_train = COCO(TRAIN_ANNOTATIONS_PATH)
coco_val = COCO(VAL_ANNOTATIONS_PATH)

# Filter only wanted categories
cat_names = ['tomato-sauce', 'avocado']
(cat_ids, cat_names, train_img_ids) = filtered_cats(coco_train, cat_names=cat_names)
(_, _, val_img_ids) = filtered_cats(coco_val, cat_names=cat_names)

# Get image relative paths
train_images = coco_train.loadImgs(train_img_ids)
train_img_paths = [img["file_name"] for img in train_images]
val_images = coco_val.loadImgs(val_img_ids)
val_img_paths = [img["file_name"] for img in val_images]
n_train_imgs = len(train_img_ids)
n_val_imgs = len(val_img_ids)

train_data_gen = DataGeneration(coco_train, SIZE_X, SIZE_Y, cat_ids)
val_data_gen = DataGeneration(coco_val, SIZE_X, SIZE_Y, cat_ids)

i = 0
image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, train_img_paths[0])
print(image_path)
x_train = train_data_gen.x_sample(image_path)
y_train = train_data_gen.y_sample(train_img_ids[0])
plt.figure()
plt.imshow(cv2.cvtColor(x_train, cv2.COLOR_BGR2RGB))
plt.imshow(y_train[:, :, i], alpha=0.5)

# Convert back to coco
mask = cocomask.encode(y_train[:, :, i])

