from PIL import Image
import numpy as np
from glob import glob
import os
from random import sample
def zip_image_seg(seg_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    all_outputs = list(map(lambda x: x.split("\\")[-1], glob(output_path + "\\" + "*")))
    all_seg_pngs = glob(seg_path + "\\" + "*")
    all_seg_pngs = sample(all_seg_pngs, len(all_seg_pngs))
    all_jpgs = list(map(lambda x: x.replace("2017_seg", "2017").replace("png", "jpg"), all_seg_pngs))

    for i in range(len(all_seg_pngs)):
        seg_png_path = all_seg_pngs[i]
        if seg_png_path.split("\\")[-1] in all_outputs:
            #print("skip : {}".format(seg_png_path))
            continue

        jpg_path = all_jpgs[i]
        try:
            seg_png = Image.open(seg_png_path).convert('L')
            jpg = Image.open(jpg_path)
        except:
            print("error skip : {}".format(seg_png_path))
            continue

        seg_png, jpg = map(lambda x: np.asarray(x), [seg_png, jpg])
        if len(jpg.shape) != 3:
            #### [3, h, w] -> [h, w, 3]
            jpg = np.transpose(np.asarray([jpg] * 3), [1, 2, 0])

        image = Image.fromarray(np.concatenate([jpg, seg_png[...,np.newaxis]], axis=-1).astype(np.uint8))
        file_tail = seg_png_path.split("\\")[-1].split(".")[0]
        image.save(r"{}\{}.png".format(output_path, file_tail))
        if i > 0 and i % 100 == 0:
            print("finish {}".format(i))

if __name__ == "__main__":
    zip_image_seg(r"E:\Temp\train2017_seg", r"E:\Temp\train2017_with_label")
    pass

