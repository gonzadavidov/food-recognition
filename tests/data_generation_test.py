# Test of DataGeneration class

import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

from dataset_filtering.data_generation import DataGeneration
from dataset_filtering.filter_cats import filtered_cats


TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"

SIZE_X = 426
SIZE_Y = 426

# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Filter 2 more frequent categories
n_cats = 2
(cat_ids, cat_names, img_ids) = filtered_cats(coco, n=n_cats)

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]

data_gen = DataGeneration(coco, SIZE_X, SIZE_Y, cat_ids)

i = 200
while True:
    annotation_ids = coco.getAnnIds(imgIds=img_ids[i], catIds=cat_ids)
    if len(annotation_ids) > 1:
        break
    i += 1
print(i)
image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img_paths[i])
x_train = data_gen.x_sample(image_path)
y_train = data_gen.y_sample(img_ids[i])
plt.figure()
plt.imshow(cv2.cvtColor(x_train, cv2.COLOR_BGR2RGB)); plt.axis('off')
annotation_ids = coco.getAnnIds(imgIds=img_ids[i], catIds=cat_ids)
annotations = coco.loadAnns(annotation_ids)
# Render annotations on top of the image
# coco.showAnns(annotations)
plt.imshow(y_train[:, :, n_cats], alpha=0.5); plt.axis('off')

# Check class weights
cat_dict = dict(zip(cat_ids, cat_names))
class_weights = data_gen.get_class_weights(img_ids[i])
for k, id in enumerate(cat_ids):
    if class_weights[k]:
        print(f"Category: {cat_dict[id]}, {class_weights[k]*100: .1f} %")

print(f"Background: {class_weights[n_cats] * 100: .1f} %")

for id in img_ids:
    class_weights += data_gen.get_class_weights(img_ids[i])

class_weights /= len(img_ids)