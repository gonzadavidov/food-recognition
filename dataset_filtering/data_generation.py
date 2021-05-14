import cv2
import numpy as np
from pycocotools import mask as cocomask
import skimage.transform as transf


class DataGeneration:
    """
    DataGeneration
    Class dedicated to generate the input and output samples for the net:
        - Input samples generated by reading the images and loading them to an array
        - Output samples generated by reading the COCO object and loading annotations into
            corresponding array channels
    """
    def __init__(self, coco, x_size, y_size, cat_ids):
        self.coco = coco
        self.x_size = x_size
        self.y_size = y_size
        self.cat_ids = cat_ids
        self.n_cats = len(cat_ids)
        self.cat_dict = dict(zip(cat_ids, range(len(cat_ids))))

    def x_sample(self, img_path):
        img = cv2.imread(img_path)
        return cv2.resize(img, (self.x_size, self.y_size))

    def y_sample(self, img_id):
        ret = np.zeros((self.x_size, self.y_size, self.n_cats))
        cat_ids = self.cat_ids
        coco = self.coco
        img = coco.loadImgs(img_id)[0]

        # Only load annotations of the wanted categories on the current image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annotations = coco.loadAnns(ann_ids)
        # Load each annotation on the corresponding channel
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            n = m.shape[2]
            union = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint32)
            # If more than one item, put all together in one channel
            for j in range(n):
                # m.shape has a shape of (height, width, 1), convert it to a shape of (height, width)
                aux = m[:, :, j].reshape((img['height'], img['width']))
                union = aux | union
            mask = transf.resize(union, (self.x_size, self.y_size))
            channel = self.cat_dict[annotation['category_id']]
            # Add mask to the corresponding channel
            ret[:, :, channel] = mask

        return ret
