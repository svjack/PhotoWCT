import tensorflow as tf
from ops import pad_reflect
import torchfile
from tensorflow.contrib.layers.python.layers import utils
import pywt.data
import numpy as np

def single_pooling_func(single_hw):
    LL, (LH, HL, HH) = pywt.dwt2(single_hw, 'haar')
    #### [4, h, w]
    return np.asarray([LL, LH, HL, HH])


def vgg_from_t7(t7_file, target_layer=None, inp = None, use_wavelet_pooling = True):
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    x = inp

    max_count = 0

    indices_list = []
    for idx,module in enumerate(t7.modules):
        name = module.name.decode() if module.name is not None else None
        if idx == 0:
            name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias

        if module._typename == b'nn.SpatialReflectionPadding':
            #x = Lambda(pad_reflect)(x)
            x = pad_reflect(x)
        elif module._typename == b'nn.SpatialConvolution':
            filters = module.nOutputPlane
            kernel_size = module.kH
            weight = module.weight.transpose([2,3,1,0])

            bias = module.bias
            x = tf.layers.conv2d(
                inputs = x,
                filters = filters,
                kernel_size = (kernel_size, kernel_size),
                strides=(1, 1),
                padding='valid',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer=tf.constant_initializer(weight),
                bias_initializer=tf.constant_initializer(bias),
                reuse=tf.AUTO_REUSE,
                name="conv2d_{}".format(name)
            )

        elif module._typename == b'nn.ReLU':
            #x = Activation('relu', name=name)(x)
            x = tf.nn.relu(x, name=name)
            tf.add_to_collection("relu_targets", x)

        elif module._typename == b'nn.SpatialMaxPooling':
            if use_wavelet_pooling:
                #x = MaxPooling2D(padding='same', name=name)(x)
                h, w, c = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
                c = int(x.get_shape()[-1])
                #### [batch * c, h, w]
                flatten_hw = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, h, w])

                #### [batch * c, 4, h, w]
                pooling_conslusion = tf.map_fn(
                    lambda x: tf.py_func(
                        single_pooling_func, inp=[x], Tout=tf.float32
                    ),
                    flatten_hw,
                    dtype=tf.float32
                )

                #### [4, batch, h, w, c]
                h_2, w_2 = map(lambda x: tf.cast(tf.divide(x, 2), tf.int32), [h, w])
                pooling_conslusion = tf.transpose(tf.reshape(tf.transpose(pooling_conslusion, [1, 0, 2, 3]), [4, -1, c, h_2, w_2]),
                                                  [0, 1, 3, 4, 2])

                LL, LH, HL, HH = tf.unstack(pooling_conslusion, axis=0)
                #### [batch, h, w, c]
                L = LL
                #### [3, batch, h, w, c]
                H = tf.stack([LH, HL, HH])
                x = L
                indices_list.append(H)
            else:
                x, indices = tf.nn.max_pool_with_argmax(
                    x, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "SAME"
                )

                x = tf.identity(x, name="x_count_{}".format(max_count))
                max_count += 1

                indices_list.append(indices)
        # elif module._typename == b'nn.SpatialUpSamplingNearest': # Not needed for VGG
        #     x = Upsampling2D(name=name)(x)
        else:
            raise NotImplementedError(module._typename)

        if name == target_layer:
            # print("Reached target layer", target_layer)
            break

    return (x, indices_list, dict(map(lambda t2: (t2[0].split("/")[-1], t2[1]),utils.convert_collection_to_dict("relu_targets").items())))


def test_():
    #### models/vgg_normalised.t7
    #### relu5_1
    inp, x, indices_list, relu_targets = vgg_from_t7(r"C:\Coding\Python\StyleLab\models\vgg_normalised.t7", "relu5_1")
    print(inp)
    print(x)
    print(indices_list)

    from pprint import pprint
    pprint(tf.trainable_variables())
    pprint(relu_targets)

if __name__ == "__main__":
    test_()

    pass