# Food Recognition Challenge

The training and evaluation of this network was done in Google Colaboratory, so it's not straightforward to run the notebooks because the paths are different.

The results are explained in the [report](./report/report.pdf) and can be visualised [by looking at the predicted outputs of the network](https://drive.google.com/drive/folders/1Te9qaGptbRpP6jC82cyaJFufIlWq-4v4?usp=sharing).

The repo is organised as follows: 
* [colab_notebooks](./colab_notebooks/) directory contains:
  * [\<model>_pretrained.ipynb](./colab_notebooks/vgg19_pretrained.ipynb) notebooks for building and training each one of the chosen backbones, using both our own implementation and the [Segmentation Models](https://github.com/qubvel/segmentation_models) Library.
  * [models_evaluator](./colab_notebooks/models_evaluator.ipynb) notebook for evaluating each model both by its metrics and looking at the predictions.
* [dataset_filtering](./dataset_filtering/) directory contains:
  * [filtered_cats and most_annotated](./dataset_filtering/__pycache__/filter_cats.cpython-37.pyc) functions used to filter the MS-COCO dataset by categories.
  * [DataGeneration](./dataset_filtering/__pycache__/data_generation.cpython-37.pyc) class dedicated to generate the input and output samples for the net.
* [model](./model/) directory contains test files regarding the construction, training and loading of Keras models. 
* [tests](./tests/) directory contains all kind of test files for the modules made, and more.
* The dataset should be added at a data/ folder from the root directory


Authors:
* [Facundo David Farall](https://github.com/ffarall)
* [Gonzalo Joaquin Davidov](https://github.com/gonzadavidov)
* [Rafael Nicolas Trozzo](https://github.com/nicotrozzo)