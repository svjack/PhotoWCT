from __future__ import division, print_function

import os
import argparse
import numpy as np
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop, get_img_ori
import scipy
import time
from wct_edit import WCT

from functools import reduce
from time import sleep

from scipy.io import loadmat
from numpy import vectorize
from PIL import Image

def colors_map():
    colors = list(map(lambda x: hash(tuple(x)),loadmat('data/color150.mat')['colors'].tolist()))
    return colors

colors_list = colors_map()

def l3_assign(l3):
    return colors_list.index(l3)

l3_assign_vec = vectorize(l3_assign)

def image_to_pixel_map(input_image):
    assert len(input_image.shape) == 3
    h, w, c = input_image.shape
    label_looked_up_array = l3_assign_vec(np.asarray(list(map(lambda l3: hash(tuple(l3)),input_image.reshape([h * w, c]).tolist()))).reshape([h, w]))
    return label_looked_up_array


parser = argparse.ArgumentParser()

parser.add_argument('--input-checkpoint', type=str, help='Path to restore model', required=True)

#parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
#parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/cpu:0')

parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=512)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=512)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=512)

parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('-r','--random', type=int, help="Choose # of random subset of images from style folder", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

args = parser.parse_args()


def main():
    start = time.time()

    # Load the WCT model
    wct_model = WCT(input_checkpoint=args.input_checkpoint,
                    relu_targets=args.relu_targets,
                    vgg_path=args.vgg_path,
                    device=args.device,
                    ss_patch_size=args.ss_patch_size,
                    ss_stride=args.ss_stride)

    print("model construct end !")

    # Get content & style full paths
    if os.path.isdir(args.content_path):
        content_files = get_files(args.content_path)
    else: # Single image file
        content_files = [args.content_path]

    content_seg_files = list(map(lambda x: x.replace("2017", "2017_seg").replace("jpg", "png"), content_files))
    assert reduce(lambda a, b : a + b, map(lambda x: int(os.path.exists(x)),content_seg_files + content_files)) > 0

    if os.path.isdir(args.style_path):
        style_files = get_files(args.style_path)
        if args.random > 0:
            style_files = np.random.choice(style_files, args.random)
    else: # Single image file
        style_files = [args.style_path]

    style_seg_files = list(map(lambda x: x.replace("2017", "2017_seg").replace("jpg", "png"), style_files))
    assert reduce(lambda a, b : a + b, map(lambda x: int(os.path.exists(x)),style_seg_files + style_files)) > 0

    os.makedirs(args.out_path, exist_ok=True)

    count = 0

    ### Apply each style to each content image
    for i in range(len(content_files)):
        content_fullpath = content_files[i]
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

        content_img = get_img(content_fullpath)
        if args.content_size > 0:
            content_img = np.asarray(Image.fromarray(content_img.astype(np.uint8)).resize((args.content_size, args.content_size))).astype(np.float32)
        content_seg = get_img_ori(content_seg_files[i])
        if args.content_size > 0:
            content_seg = np.asarray(Image.fromarray(content_seg.astype(np.uint8)).resize((args.content_size, args.content_size))).astype(np.float32)
            content_seg = image_to_pixel_map(content_seg)[...,np.newaxis]
        content_img = np.concatenate([content_img, content_seg], axis=-1)

        for i in range(len(style_files)):
            style_fullpath = style_files[i]
            style_prefix, _ = os.path.splitext(style_fullpath)
            style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

            style_img = get_img(style_fullpath)
            style_seg = get_img_ori(style_seg_files[i])

            if args.style_size > 0:
                style_img = resize_to(style_img, args.style_size)
                style_seg = resize_to(style_seg, args.style_size)

            if args.crop_size > 0:
                style_img = center_crop(style_img, args.crop_size)
                style_seg = center_crop(style_seg, args.crop_size)

            style_seg = image_to_pixel_map(style_seg)[...,np.newaxis]
            style_img = np.concatenate([style_img, style_seg], axis=-1)

            assert not args.keep_colors
            if args.keep_colors:
                style_img = preserve_colors_np(style_img, content_img)

            # if args.noise:  # Generate textures from noise instead of images
            #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
            #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            # Run the frame through the style network
            stylized_rgb = wct_model.predict(content_img, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)


            if args.passes > 1:
                for _ in range(args.passes-1):
                    stylized_rgb = np.concatenate([stylized_rgb ,content_seg], axis=-1)
                    stylized_rgb = wct_model.predict(stylized_rgb, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)

            # Stitch the style + stylized output together, but only if there's one style image
            assert not args.concat
            if args.concat:
                # Resize style img to same height as frame
                style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            ####+++++++++++++++++++++++++++++++++++

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
            save_img(out_f, stylized_rgb)

            count += 1
            print("{}: Wrote stylized output image to {}".format(count, out_f))

    print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))


if __name__ == '__main__':
    main()
