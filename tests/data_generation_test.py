# Test of DataGeneration class

import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from dataset_filtering.data_generation import DataGeneration
from dataset_filtering.filter_cats import filtered_cats


TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"

SIZE_X = 426
SIZE_Y = 426

# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Filter 16 more frequent categories
n_cats = 16
(cat_ids, cat_names, img_ids) = filtered_cats(coco, n=n_cats)

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]

data_gen = DataGeneration(coco, SIZE_X, SIZE_Y, cat_ids)

i = 0
image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img_paths[0])
x_train = data_gen.x_sample(image_path)
y_train = data_gen.y_sample(img_ids[0])
plt.imshow(x_train)
plt.imshow(y_train[:, :, 1], alpha=0.5)

