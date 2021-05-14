from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import random
from model import build_vgg19_unet


if __name__ == "__main__":
    SIZE = 426
    input_shape = (SIZE, SIZE, 3)
    n_classes = 16

    train_image_directory = 'train/images/'
    train_mask_directory = 'train/masks/'

    X_train = []
    Y_train = []

    images = os.listdir(train_image_directory)
    for i, image_name in enumerate(images):
        if(image_name.split('.')[1] == 'jpg'):
            image = cv2.imread(train_image_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            X_train.append(np.array(image))

    masks = os.listdir(train_mask_directory)
    for i, mask_name in enumerate(masks):
        if(mask_name.split('.')[1] == 'jpg'):
            mask = cv2.imread(train_mask_directory+mask_name, 0)
            mask = Image.fromarray(mask)
            mask = mask.resize((SIZE, SIZE))
            Y_train.append(np.array(mask))


    X_train = np.expand_dims(X_train, axis=3)
    X_train = normalize(X_train, axis=1)
    Y_train = np.expand_dims(np.array(Y_train), axis=3) / 255.  #change the 255 if masks already normalized

    print("Class values in dataset are:", np.unique(Y_train))  #Check

    train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))  #Check this line

    #Sanity check for images and masks
    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(Y_train[image_number], (256, 256)), cmap='gray')

    #Create Model
    model = build_vgg19_unet(input_shape, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #Check this line
    model.summary()

    history = model.fit(X_train, y_train_cat, 
                        batch_size=16,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_test, y_test_cat),
                        #class_weight=class_weights,
                        shuffle=False)
    
    model.save('food_recognition_50_epochs_test.hdf5') #Saving a model with 50 epochs

