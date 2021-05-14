from pycocotools.coco import COCO
import pandas as pd


def most_annotated(c, n):
    """
     most_annotated:

     @brief     Returns a for the 'n' most annotated categories its id, name and amount of annotations
     @param c   Coco object instance
     @param n   Amount of categories desired
    """
    ids = c.getCatIds()
    df = pd.DataFrame(columns=['name', 'id', 'freq'])
    for id in ids:
        df = df.append({'name': c.loadCats(id)[0]['name'], 'id': id, 'freq': len(c.getImgIds(catIds=id))}, ignore_index=True)
    df = df.sort_values(by=['freq'], ascending=False)
    top_n = df.head(n)

    return top_n['id'].tolist(), top_n['name'].tolist(), top_n['freq'].tolist()


def filtered_cats(coco: COCO, cat_names: list = [], n: int = 0):
    """
     filtered_cats:

     @brief     For the specified filtering, returns lists of COCO categories ids and image ids
                The filtering can be made either by:
                    - Asking for the 'n' most annotated categories: n>0
                    - Specifying each category name: cat_names=[list of names]

     @param coco            Instance of the coco object pointing to the whole dataset
     @param n               Amount of categories requested if most annotated are desired
     @param cat_names       List of category names if specific categories are desired, ex: ['water', 'pear']

     @return cat_ids        List of category ids matching the filtering
     @return cat_names      List of each category's name
     @return img_ids        List of image ids annotated with the filtered categories

    """

    # Obtain category ids for each case
    if n > 0:
        (cat_ids, cat_names, _) = most_annotated(coco, n)
    else:
        cat_ids = coco.getCatIds(catNms=cat_names)
        cats = coco.loadCats(cat_ids)
        cat_names = [cat['name'] for cat in cats]
    # Generate a list of all `image_ids` that match the wanted categories
    img_ids = [coco.getImgIds(catIds=[id]) for id in cat_ids]
    img_ids = [item for sublist in img_ids for item in sublist]  # List of lists flattening

    return cat_ids, cat_names, img_ids
