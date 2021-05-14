import os
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import skimage.io as io
import skimage.transform as transf
import skimage
import time
import numpy as np
from tqdm import tqdm

from dataset_filtering.filter_cats import filtered_cats

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"

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
# Input training set will be composed by all images in this small test
batch_size = n_imgs

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]

print(f'Starting to read images, about to do {n_imgs} iterations...')
x_train = np.zeros((SIZE_X, SIZE_Y, 3, batch_size))
for k, rel_path in tqdm(enumerate(img_paths)):
    # Save images to the target directory of filtered data
    image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, rel_path)
    img = io.imread(image_path)
    img = transf.resize(img, (SIZE_X, SIZE_Y))
    x_train[:, :, :, k] = img

# Create dictionary with cat_id as key and a number from 0 to n_cats - 1 as value
cat_dict = dict(zip(cat_ids, range(n_cats)))
# Output training set
y_train = np.zeros((SIZE_X, SIZE_Y, n_cats, batch_size))
print(f'Generating images masks, other {n_imgs} iterations...')
# Now for each image, obtain it's corresponding masks
for k, img_id in tqdm(enumerate(img_ids)):
    # Only load annotations of the wanted categories on the selected
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    annotations = coco.loadAnns(ann_ids)
    # Load each annotation on the corresponding channel
    for _idx, annotation in enumerate(annotations):
        rle = cocomask.frPyObjects(annotation['segmentation'], images[k]['height'], images[k]['width'])
        m = cocomask.decode(rle)
        n = m.shape[2]
        union = np.zeros((m.shape[0], m.shape[1]), dtype=np.int32)
        for j in range(n):
            # m.shape has a shape of (height, width, 1), convert it to a shape of (height, width)
            aux = m[:, :, j].reshape((images[k]['height'], images[k]['width']))
            union = aux | union
        mask = transf.resize(union, (SIZE_X, SIZE_Y))
        channel = cat_dict[annotation['category_id']]

        y_train[:, :, channel, k] = mask

