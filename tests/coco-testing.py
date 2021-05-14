from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import pandas as pd
import time

from dataset_filtering.filter_cats import most_annotated

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"
# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Filter wanted categories
most_frequent = True
if most_frequent:
    n_cats = 16
    (cat_ids, cat_names, _) = most_annotated(coco, n_cats)

else:
    cat_names = ['rice']
    n_cats = len(cat_names)
    cat_ids = coco.getCatIds(catNms=cat_names)
categories = coco.loadCats(cat_ids)

# This generates a list of all `image_ids` that match the wanted categories
image_ids = [coco.getImgIds(catIds=[id]) for id in cat_ids]
image_ids = [item for sublist in image_ids for item in sublist] # List of lists flattening

# For this demonstration, we will randomly choose an image_id
#random_image_id = random.choice(image_ids)
random_image_id = 94213

# Now that we have an image_id, we can load its corresponding object by doing :
img = coco.loadImgs(random_image_id)[0]

image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
I = io.imread(image_path)

annotation_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
annotations = coco.loadAnns(annotation_ids)

# load and render the image
plt.figure()
plt.imshow(I); plt.axis('off')
# Render annotations on top of the image
coco.showAnns(annotations)

# Convert segmentation to pixel level mask
pylab.rcParams['figure.figsize'] = (10, 40.0)
plt.figure()
output_mask = np.zeros((img['height'], img['width'], n_cats))
cat_dict = dict(zip(cat_ids, zip(range(n_cats), cat_names)))
for _idx, annotation in enumerate(annotations):
    plt.subplot(len(annotations), 1, _idx+1, title=cat_dict[annotation['category_id']][1])
    rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
    m = cocomask.decode(rle)
    n = m.shape[2]
    union = np.zeros((m.shape[0], m.shape[1]), dtype=np.int32)
    for k in range(n):
        # m.shape has a shape of (300, 300, 1), convert it to a shape of (300, 300)
        aux = m[:, :, k].reshape((img['height'], img['width']))
        union = aux | union
    plt.imshow(union)
    channel = cat_dict[annotation['category_id']][0]
    output_mask[:, :, channel] = union

print('Sleeping')
time.sleep(5)
print(output_mask.shape)
plt.figure()
n_rows = 4
for cat_id in cat_ids:
    channel = cat_dict[cat_id][0]
    plt.subplot(n_cats / n_rows, n_rows, channel+1, title=cat_dict[cat_id][1])
    plt.imshow(output_mask[:, :, channel])


