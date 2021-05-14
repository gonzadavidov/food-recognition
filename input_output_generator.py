# %%
from pycocotools.coco import COCO
from dataset_utils.filter_cats import filtered_cats
import skimage.transform as transf
import skimage.io as io
import numpy as np
from tqdm import tqdm

# %%
TRAIN_IMAGES_DIRECTORY = "dataset_utils/data/train/images"
TRAIN_ANNOTATIONS_PATH = "dataset_utils/data/train/annotations.json"

# Build coco object pointing to the annotations
coco = COCO(TRAIN_ANNOTATIONS_PATH)

# Filter 16 more frequent categories
n_cats = 16
(cat_ids, cat_names, img_ids) = filtered_cats(coco, n=n_cats)

# Get image relative paths
images = coco.loadImgs(img_ids)
img_paths = [img["file_name"] for img in images]


# %%
SIZE_X = 426
SIZE_Y = 426

train_images = []

for img_path in tqdm(img_paths):
    img_path = TRAIN_IMAGES_DIRECTORY + "/" + img_path
    img = io.imread(img_path)
    img = transf.resize(img, (SIZE_X, SIZE_Y))
    train_images.append(img)

x_train = np.array(train_images)

# %%
INPUT_ARRAY_PATH = "train-v0.4/train/images/input_array.npy"
np.save(INPUT_ARRAY_PATH, x_train)
