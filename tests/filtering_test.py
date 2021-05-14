# Test of dataset filtering and image showing

import os
from pycocotools.coco import COCO
import skimage.io as io
import time
import numpy as np

from dataset_filtering.filter_cats import filtered_cats

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"
FILTERED_TRAIN_IMAGES_DIRECTORY = "filtered_data/train/images"

# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Filter 16 more frequent categories
n_cats = 16
(cat_ids, cat_names, img_ids) = filtered_cats(coco, n=n_cats)
# Only obtain images annotated with water or pears
# cat_names = ['water', 'pear']
# (cat_ids, cat_names, img_ids) = filtered_cats(coco, cat_names=cat_names)
# n_cats = len(cat_names)

# Print filtered categories
print(f'{n_cats} categories, with {len(img_ids)} images')

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]

print('Starting to read')
start_time = time.time()
for rel_path in img_paths:
    # Save images to the target directory of filtered data
    image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, rel_path)
    I = io.imread(image_path)
    io.imsave(FILTERED_TRAIN_IMAGES_DIRECTORY + rel_path, I)
    # np.save(FILTERED_TRAIN_IMAGES_DIRECTORY + rel_path[:-4], I)
print(f'Finished reading, elapsed time: {round(time.time() - start_time, 2)} seconds')

# Now for each image, obtain it's corresponding masks
# Create dictionary with cat_id as key and a number from 0 to n_channels - 1 as value
cat_dict = dict(zip(cat_ids, range(n_cats)))
for id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=id, catIds=cat_ids)
    annotations = coco.loadAnns(ann_ids)
