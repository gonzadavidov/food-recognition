import os
from pycocotools.coco import COCO
from filter_cats import filtered_cats

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
