from collections import defaultdict, OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K



def build_vgg16_unet(input_shape, n_classes=1):
    inputss = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputss)
    vgg16.trainable = False


    layer_size_dict = defaultdict(list)
    inputs = []
    for lay_idx, c_layer in enumerate(vgg16.layers):
        if not c_layer.__class__.__name__ == 'InputLayer':
            layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
        else:
            inputs += [c_layer]
    # freeze dict
    layer_size_dict = OrderedDict(layer_size_dict.items())
    for k,v in layer_size_dict.items():
        print(k, [w.__class__.__name__ for w in v])

    # take the last layer of each shape and make it into an output
    pretrained_encoder = Model(inputs = vgg16.get_input_at(0), 
                              outputs = [v[-1].get_output_at(0) for k, v in layer_size_dict.items()])
    pretrained_encoder.trainable = False
    n_outputs = pretrained_encoder.predict([t0_img])
    for c_out, (k, v) in zip(n_outputs, layer_size_dict.items()):
        print(c_out.shape, 'expected', k)


    x_wid, y_wid = t0_img.shape[1:3]
    in_t0 = Input(t0_img.shape[1:], name = 'T0_Image')
    wrap_encoder = lambda i_layer: {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}

    t0_outputs = wrap_encoder(in_t0)
    lay_dims = sorted(t0_outputs.keys(), key = lambda x: x[0])
    skip_layers = 2
    last_layer = None
    for k in lay_dims[skip_layers:]:
        cur_layer = t0_outputs[k]
        channel_count = cur_layer._keras_shape[-1]
        cur_layer = Conv2D(channel_count//2, kernel_size=(3,3), padding = 'same', activation = 'linear')(cur_layer)
        cur_layer = BatchNormalization()(cur_layer) # gotta keep an eye on that internal covariant shift
        cur_layer = Activation('relu')(cur_layer)
        
        if last_layer is None:
            x = cur_layer
        else:
            last_channel_count = last_layer._keras_shape[-1]
            x = Conv2D(last_channel_count//2, kernel_size=(3,3), padding = 'same')(last_layer)
            x = UpSampling2D((2, 2))(x)
            x = concatenate([cur_layer, x])
        last_layer = x
    final_output = Conv2D(n_classes, kernel_size=(1,1), padding = 'same', activation = 'sigmoid')(last_layer)
    crop_size = 20
    final_output = Cropping2D((crop_size, crop_size))(final_output)
    final_output = ZeroPadding2D((crop_size, crop_size))(final_output)
    unet_model = Model(inputs = [in_t0],
                      outputs = [final_output], name="VGG16_U-Net")
    return unet_model


SIZE_X = 128
SIZE_Y = 128
WANTED_CATS = ['egg', 'rice']
N_CATS = len(WANTED_CATS)

INPUT_SHAPE = (SIZE_X, SIZE_Y, 3)
BATCH_SIZE = 32

model = build_vgg16_unet(INPUT_SHAPE, N_CATS)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.summary()