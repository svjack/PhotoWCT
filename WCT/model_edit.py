import tensorflow as tf
from vgg_normalised import vgg_from_t7
from ops import Conv2DReflect, torch_decay, wct_tf, wct_style_swap, adain, unpooling
from keras.layers import UpSampling2D
from collections import namedtuple
from functools import partial, reduce
import pywt.data

mse = tf.losses.mean_squared_error
clip = lambda x: tf.clip_by_value(x, 0, 1)

#### [4, h, w] -> [h1, w1]
def single_unpooling_func(single_hw):
    LL = single_hw[0]
    LH = single_hw[1]
    HL = single_hw[2]
    HH = single_hw[3]

    return pywt.idwt2((LL, (LH, HL, HH)), "haar")


#### valid fields decoded content_input
EncoderDecoder = namedtuple('EncoderDecoder',
                            'content_input content_encoded \
                             style_encoded \
                             decoder_input, decoded decoded_encoded \
                             pixel_loss feature_loss tv_loss total_loss \
                             train_op learning_rate global_step \
                             summary_op')

#### first resize second one-hot
def resize_mask(mask, to_h, to_w):
    #### mask [batch, h, w, 1]
    assert len(mask.get_shape()) == 4
    #### must use nn
    return tf.image.resize_images(mask, (to_h, to_w), method=1)

def stacklized_one_hot(mask, channel_dim):
    #### [batch, h, w, 1] -> [batch, h, w, 150] -> [batch, h, w, 150, 3] -> [150, batch, h, w, 3]
    return tf.transpose(tf.tile(tf.expand_dims(tf.one_hot(tf.squeeze(tf.cast(mask, tf.int32), [-1]), depth=150), axis=-1), [1, 1, 1, 1, channel_dim]), [3, 0, 1, 2, 4])

def feature_map_lookedup(feature_map ,mask):
    to_h, to_w, c_dim = tf.shape(feature_map)[1], tf.shape(feature_map)[2], tf.shape(feature_map)[3]
    if len(mask.get_shape()) == 3:
        mask = tf.expand_dims(mask, -1)

    #### [150, batch, to_h, to_w, c_dim]
    r_mask = resize_mask(mask, to_h, to_w)
    stacked_one_hot = stacklized_one_hot(r_mask, c_dim)
    one_hot_list = tf.unstack(stacked_one_hot, axis=0)
    return (list(map(lambda one_hot_tensor: tf.cast(one_hot_tensor, tf.float32) * feature_map, one_hot_list)), one_hot_list)

class WCTModel(object):
    def __init__(self, mode='train', relu_targets=['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1'], vgg_path=None,
                 use_wavelet_pooling = False,
                 *args, **kwargs):
        '''
            Args:
                mode: 'train' or 'test'. If 'train' then training & summary ops will be added to the graph
                relu_targets: List of relu target layers corresponding to decoder checkpoints
                vgg_path: Normalised VGG19 .t7 path
        '''
        self.mode = mode
        self.vgg_path = vgg_path
        self.relu_targets = relu_targets

        #### 3 -> 4
        #self.style_input = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3), name='style_img')
        #self.style_input = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.,0.]]]]), shape=(None, None, None, 4), name='style_img')
        self.style_input = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.,0., 0.]]]]), shape=(None, None, None, 5), name='style_img')

        self.use_wavelet_pooling = use_wavelet_pooling

        self.alpha = tf.placeholder_with_default(1., shape=[], name='alpha')

        # Style swap settings
        self.swap5 = tf.placeholder_with_default(tf.constant(False), shape=[])
        self.ss_alpha = tf.placeholder_with_default(.7, shape=[], name='ss_alpha')

        # Flag to use AdaIN instead of WCT
        self.use_adain = tf.placeholder_with_default(tf.constant(False), shape=[])

        self.encoder_decoders = []

        if self.mode == "train":
            style_encodings = [None]
        else:
            #with tf.name_scope("style_encoder"):
            with tf.variable_scope("vgg_encoder", reuse=tf.AUTO_REUSE):
                deepest_target = sorted(relu_targets)[-1]
                self.deepest_target = deepest_target
                style_input = self.style_input[...,:-1]
                style_mask = self.style_input[...,-1]
                self.style_mask = style_mask

                vgg_outputs, indices_list, relu_targets_dict = vgg_from_t7(vgg_path, target_layer=deepest_target, inp=style_input,
                                                                           use_wavelet_pooling=self.use_wavelet_pooling)
                style_encoding_layers = [relu_targets_dict[relu] for relu in relu_targets]
                style_encodings = style_encoding_layers

        for i, (relu, style_encoded) in enumerate(zip(relu_targets, style_encodings)):
            print('Building encoder/decoder for relu target', relu)

            if i == 0:
                # Input tensor will be a placeholder for the first encoder/decoder
                input_tensor = None
            else:
                # Input to intermediate levels is the output from previous decoder
                input_tensor = clip(self.encoder_decoders[-1].decoded)


            enc_dec = self.build_model(relu, input_tensor=input_tensor, style_encoded_tensor=style_encoded,
                                       encoder_indices=None, **kwargs)

            self.encoder_decoders.append(enc_dec)


    def build_model(self,
                    relu_target,
                    input_tensor,
                    style_encoded_tensor=None,
                    batch_size=8,
                    feature_weight=1,
                    pixel_weight=1,
                    tv_weight=1.0,
                    learning_rate=1e-4,
                    lr_decay=5e-5,
                    ss_patch_size=3,
                    ss_stride=1,
                    encoder_indices = None):
        '''Build the EncoderDecoder architecture for a given relu layer.

            Args:
                relu_target: Layer of VGG to decode from
                input_tensor: If None then a placeholder will be created, else use this tensor as the input to the encoder
                style_encoded_tensor: Tensor for style image features at the same relu layer. Used only at test time.
                batch_size: Batch size for training
                feature_weight: Float weight for feature reconstruction loss
                pixel_weight: Float weight for pixel reconstruction loss
                tv_weight: Float weight for total variation loss
                learning_rate: Float LR
                lr_decay: Float linear decay for training
            Returns:
                EncoderDecoder namedtuple with input/encoding/output tensors and ops for training.
        '''
        with tf.name_scope('encoder_decoder_'+relu_target):
            ### Build encoder for reluX_1
            #with tf.name_scope('content_encoder_'+relu_target):
            with tf.variable_scope("vgg_encoder", reuse=tf.AUTO_REUSE):
                if input_tensor is None:
                    # This is the first level encoder that takes original content imgs
                    #### 3 -> 4
                    #content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3), name='content_imgs')
                    #content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.,0.]]]]), shape=(None, None, None, 4), name='content_imgs')
                    content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.,0.,0.]]]]), shape=(None, None, None, 5), name='content_imgs')
                else:
                    # This is an intermediate-level encoder that takes output tensor from previous level as input
                    content_imgs = input_tensor

                deepest_target = sorted(self.relu_targets)[-1]
                self.deepest_target = deepest_target
                vgg_inputs = content_imgs[...,:-1]
                content_mask = content_imgs[...,-1]

                #vgg_inputs = tf.zeros(shape=[8, 512, 512, 3], dtype=tf.float32)
                vgg_outputs, indices_list, relu_targets_dict = vgg_from_t7(self.vgg_path, target_layer=deepest_target, inp=vgg_inputs,
                                                                           use_wavelet_pooling=self.use_wavelet_pooling)


            content_layer = relu_targets_dict[relu_target]
            content_encoded = content_layer
            encoder_indices = indices_list

            ### Build style encoder & WCT if test mode
            if self.mode != 'train':
                with tf.name_scope('wct_'+relu_target):
                    assert relu_target != "relu5_1"
                    if relu_target == 'relu5_1':
                        # Apply style swap on relu5_1 encodings if self.swap5 flag is set
                        # Use AdaIN as transfer op instead of WCT if self.use_adain is set
                        # Otherwise perform WCT
                        decoder_input = tf.case([(self.swap5, lambda: wct_style_swap(content_encoded,
                                                                                     style_encoded_tensor,
                                                                                     self.ss_alpha,
                                                                                     ss_patch_size,
                                                                                     ss_stride)),
                                                 (self.use_adain, lambda: adain(content_encoded, style_encoded_tensor, self.alpha))],
                                                default=lambda: wct_tf(content_encoded, style_encoded_tensor, self.alpha))
                    else:
                        content_encoded_list, content_one_hot_list = feature_map_lookedup(content_encoded, mask=content_mask)

                        tf.identity(tf.stack(content_encoded_list, axis=0), name="content_encoded")
                        tf.identity(tf.stack(content_one_hot_list, axis=0), name="content_one_hot")

                        style_encoded_list, style_one_hot_list = feature_map_lookedup(style_encoded_tensor, mask=self.style_mask)

                        tf.identity(tf.stack(style_encoded_list, axis=0), name="style_encoded")
                        tf.identity(tf.stack(style_one_hot_list, axis=0), name="style_one_hot")

                        decoder_input_list = []
                        for i in range(len(content_encoded_list)):
                            single_content_encoded = content_encoded_list[i]
                            single_style_encoded_tensor = style_encoded_list[i]
                            single_decoder_input = tf.cond(self.use_adain,
                                                    lambda: adain(single_content_encoded, single_style_encoded_tensor, self.alpha),
                                                    lambda: wct_tf(single_content_encoded, single_style_encoded_tensor, self.alpha))
                            #### [batch, to_h, to_w, cdim]
                            single_content_mask = tf.tile(tf.image.resize_images(content_one_hot_list[i][...,-2:-1], (tf.shape(single_decoder_input)[1],
                                                                                                              tf.shape(single_decoder_input)[2]), method=1),
                                                          [1, 1, 1, tf.shape(single_decoder_input)[3]])

                            decoder_input_list.append(single_decoder_input * tf.cast(single_content_mask, tf.float32))

                        decoder_input = reduce(lambda a, b: a + b, decoder_input_list)

            else: # In train mode we're trying to reconstruct from the encoding, so pass along unchanged
                decoder_input = content_encoded

            ### Build decoder
            with tf.name_scope('decoder_'+relu_target):
                n_channels = content_encoded.get_shape()[-1].value
                Bc, Hc, Wc, Cc = tf.unstack(tf.shape(decoder_input))
                decoder_input = tf.reshape(decoder_input, [Bc, Hc, Wc, n_channels ])
                decoder_input_wrapped, decoded = self.build_decoder(decoder_input ,input_shape=(None, None, n_channels), relu_target=relu_target, encoder_indices = encoder_indices,
                                                                    use_wavelet_pooling=self.use_wavelet_pooling)


            # Content layer encoding for stylized out
            with tf.variable_scope("vgg_encoder", reuse=tf.AUTO_REUSE):
                #### should add seg into decoded
                seg_input = content_imgs[...,-2:-1]
                decoded_input = tf.concat([decoded, seg_input], axis=-1)
                decoded_encoded, _, _ = vgg_from_t7(self.vgg_path, target_layer=self.deepest_target,
                                                    inp=decoded_input, use_wavelet_pooling=self.use_wavelet_pooling)

        if self.mode == 'train':  # Train & summary ops only needed for training phase
            ### Losses
            with tf.name_scope('losses_'+relu_target):
                # Feature loss between encodings of original & reconstructed

                feature_loss = feature_weight * mse(decoded_encoded, content_encoded)

                content_imgs_sliced = content_imgs
                if int(content_imgs.get_shape()[-1]) != 3:
                    content_imgs_sliced = content_imgs[...,:3]
                # Pixel reconstruction loss between decoded/reconstructed img and original
                pixel_loss = pixel_weight * mse(decoded, content_imgs_sliced)

                # Total Variation loss
                if tv_weight > 0:
                    tv_loss = tv_weight * tf.reduce_mean(tf.image.total_variation(decoded))
                else:
                    tv_loss = tf.constant(0.)

                total_loss = 1.0 * feature_loss + 1.0 * pixel_loss + tv_loss

            with tf.name_scope('train_'+relu_target):
                global_step = tf.Variable(0, name='global_step_train', trainable=False)
                # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, 0.96, staircase=False)
                learning_rate = torch_decay(learning_rate, global_step, lr_decay)
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
                d_vars = [var for var in tf.trainable_variables() if 'vgg_encoder' not in var.name]

                train_op = d_optimizer.minimize(total_loss, var_list=d_vars, global_step=global_step)

            ### Loss & image summaries
            with tf.name_scope('summary_'+relu_target):
                feature_loss_summary = tf.summary.scalar('feature_loss', feature_loss)
                pixel_loss_summary = tf.summary.scalar('pixel_loss', pixel_loss)
                tv_loss_summary = tf.summary.scalar('tv_loss', tv_loss)
                total_loss_summary = tf.summary.scalar('total_loss', total_loss)

                content_imgs_summary = tf.summary.image('content_imgs', content_imgs_sliced)
                decoded_images_summary = tf.summary.image('decoded_images', clip(decoded))

                for var in d_vars:
                    tf.summary.histogram(var.op.name, var)

                summary_op = tf.summary.merge_all()
        else:
            # For inference set unnneeded ops to None
            pixel_loss, feature_loss, tv_loss, total_loss, train_op, global_step, learning_rate, summary_op = [None]*8

        # Put it all together
        encoder_decoder = EncoderDecoder(content_input=content_imgs,
                                         content_encoded=content_encoded,
                                         style_encoded=style_encoded_tensor,
                                         decoder_input=decoder_input,
                                         decoded=decoded,
                                         decoded_encoded=decoded_encoded,
                                         pixel_loss=pixel_loss,
                                         feature_loss=feature_loss,
                                         tv_loss=tv_loss,
                                         total_loss=total_loss,
                                         train_op=train_op,
                                         global_step=global_step,
                                         learning_rate=learning_rate,
                                         summary_op=summary_op)

        return encoder_decoder

    def build_decoder(self, code, input_shape, relu_target, encoder_indices = None, use_wavelet_pooling = True):
        '''Build the decoder architecture that reconstructs from a given VGG relu layer.

            Args:
                input_shape: Tuple of input tensor shape, needed for channel dimension
                relu_target: Layer of VGG to decode from
        '''

        decoder_num = dict(zip(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], range(1,6)))[relu_target]

        # Dict specifying the layers for each decoder level. relu5_1 is the deepest decoder and will contain all layers

        decoder_archs = {
            5: [ #    layer    filts      HxW  / InC->OutC
                (Conv2DReflect, 512),  # 16x16 / 512->512
                (UpSampling2D, 16, 512),       # 16x16 -> 32x32
                (Conv2DReflect, 512),  # 32x32 / 512->512
                (Conv2DReflect, 512),  # 32x32 / 512->512
                (Conv2DReflect, 512)], # 32x32 / 512->512
            4: [
                (Conv2DReflect, 256),  # 32x32 / 512->256
                (UpSampling2D, 32, 256),       # 32x32 -> 64x64
                (Conv2DReflect, 256),  # 64x64 / 256->256
                (Conv2DReflect, 256),  # 64x64 / 256->256
                (Conv2DReflect, 256)], # 64x64 / 256->256
            3: [
                (Conv2DReflect, 128),  # 64x64 / 256->128
                (UpSampling2D, 64, 128),       # 64x64 -> 128x128
                (Conv2DReflect, 128)], # 128x128 / 128->128
            2: [
                (Conv2DReflect, 64),   # 128x128 / 128->64
                (UpSampling2D, 128, 64)],      # 128x128 -> 256x256
            1: [
                (Conv2DReflect, 64)]   # 256x256 / 64->64
        }

        #code = Input(shape=input_shape, name='decoder_input_'+relu_target)
        #code = tf.placeholder(shape=(None,) + input_shape, name='decoder_input_'+relu_target, dtype=tf.float32)
        x = code

        ### Work backwards from deepest decoder # and build layer by layer
        decoders = list(reversed(range(1, decoder_num+1)))
        count = 0
        for d in decoders:
            for layer_tup in decoder_archs[d]:
                # Unique layer names are needed to ensure var naming consistency with multiple decoders in graph
                layer_name = '{}_{}'.format(relu_target, count)
                if layer_tup[0] == Conv2DReflect:
                    x = Conv2DReflect(x ,layer_name, filters=layer_tup[1], kernel_size=(3, 3), padding='valid', activation=tf.nn.relu)
                elif layer_tup[0] == UpSampling2D:
                    #if d in [5, 4, 3, 2]:
                    #if d in [3, 2]:
                    if d in []:
                        hw = layer_tup[1]
                        x = tf.image.resize_images(x, size=(hw * 2, hw * 2))
                    else:
                        if use_wavelet_pooling:
                            hw = layer_tup[1]
                            c = layer_tup[2]

                            indice_list = list(filter(lambda ind: int(ind.get_shape()[-1]) == int(x.get_shape()[-1]),encoder_indices))
                            assert len(indice_list) == 1
                            indice = indice_list[0]
                            indice = tf.identity(indice, name="indice_{}_{}_{}".format(hw, hw, c))
                            h, w, c = (tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3])
                            c = int(x.get_shape()[-1])

                            #### [batch, h, w, c]
                            LL = x
                            #### [4, batch, h, w, c]
                            LLHH = tf.concat([tf.expand_dims(LL, 0), indice], axis=0)

                            #### [batch, c, 4, h, w] -> [batch * c, 4, h, w]
                            flatten_4hw = tf.reshape(tf.transpose(LLHH, [1, 4, 0, 2, 3]), [-1, 4, h, w])

                            #### [batch * c, h1, w1]
                            unpooling_x = tf.map_fn(
                                lambda x: tf.py_func(
                                    single_unpooling_func,
                                    inp=[x], Tout=tf.float32
                                ),
                                flatten_4hw,
                                dtype=tf.float32
                            )

                            h1, w1 = tf.shape(unpooling_x)[1], tf.shape(unpooling_x)[2]
                            x = tf.transpose(tf.reshape(unpooling_x, [-1, c, h1, w1]), [0, 2, 3, 1])

                            x_pre = tf.image.resize_images(LL, size = (h1, w1))
                            x = tf.layers.conv2d(
                                tf.concat([x, x_pre,], axis=-1), filters=int(int(x.get_shape()[-1])),
                                kernel_size=(3, 3), padding="SAME"
                            )
                        else:
                            hw = layer_tup[1]
                            c = layer_tup[2]
                            indice_list = list(filter(lambda ind: int(ind.get_shape()[-1]) == int(x.get_shape()[-1]),encoder_indices))
                            assert len(indice_list) == 1
                            indice = indice_list[0]
                            indice = tf.identity(indice, name="indice_{}_{}_{}".format(hw, hw, c))
                            h, w = (tf.shape(x)[1], tf.shape(x)[2])
                            x_pre = tf.image.resize_images(x, size = (h * 2, w * 2))
                            x = partial(unpooling, h = h, w = w, c = c)([x_pre, indice])

                            x = tf.layers.conv2d(
                                tf.concat([x, x_pre], axis=-1), filters=int(int(x.get_shape()[-1])),
                                kernel_size=(1, 1), padding="SAME"
                            )


                count += 1

        layer_name = '{}_{}'.format(relu_target, count)

        output = Conv2DReflect(x ,layer_name, filters=3, kernel_size=(3, 3), padding='valid', activation=None)

        return (code ,output)

def test_():
    WCTModel(vgg_path = r"C:\Coding\Python\StyleLab\models\vgg_normalised.t7")

if __name__ == "__main__":
    test_()

    pass