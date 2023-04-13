<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">PhotoWCT</h3>

  <p align="center">
  Unofficial implementation of "A Closed-form Solution to Photorealistic Image Stylization"
    <br />
  </p>
</p>

[中文介绍](README.md)

## Brief introduction
This Project is the Unofficial implementation of "A Closed-form Solution to Photorealistic Image Stylization" which is a fork from the project https://github.com/eridgd/WCT-TF <br/>

The official implementation is located in https://github.com/NVIDIA/FastPhotoStyle and this project follow the TUTORIAL.md Example 3 steps in it.<br/>

Paper Link: https://arxiv.org/abs/1802.06474 <br/>

## Construction Steps
### Data Prepare
* 1 download COCO dataset from http://cocodataset.org/ (this use coco 2017)
     download vgg_normalized.t7 from web (download_vgg.sh)/<br>

* 2 This step may not be necessary if you not want to use segmentation as prior in input (with may require more times in wavelet pooling setting, set use_wavelet_pooling = True in WCTModel construction).

     use project https://github.com/CSAILVision/semantic-segmentation-pytorch to generate some segementations <br/>
     for coco dataset, if you have some confusions of use weights, you can appeal to my issue comment in https://github.com/CSAILVision/semantic-segmentation-pytorch/issues/116
     Then you can edit dataloader in test.py to generate your own segmentations, and save them in "path/to/val2017".replace("2017", "2017_seg") or "path/to/train2017".replace("2017", "2017_seg") <br/>

* 3	Use merge_pics_segs/merge.py to merge pics and segmentations, save them in "path/to/val2017".replace("2017", "2017_with_label") or "path/to/train2017".replace("2017", "2017_with_label") <br/>

### Train and Evaluate step

The merged conclusion can seen in path/to/simple2017/style_with_label.png or path/to/simple2017/content_with_label.png <br>

* 4 Train model: <br><br>
    ```bash
    python train_edit.py --relu-target relu3_1 --content-path path/to/train2017_with_label --batch-size 8 --feature-weight 1 --pixel-weight 1 --tv-weight 0.0 --learning-rate 1e-4 --max-iter 50000 --val-path path/to/val2017_with_label --checkpoint path/to/ckpt/
    ```

* 5 Evaluate the conclusion:<br><br>
   ```bash
   python stylize.py --input-checkpoint C:\Coding\Python\PhotoWCT\ckpt\model.ckpt-4601 --relu-targets relu3_1 --alpha 0.8 --style-path path/to/simple2017/style_with_label.png --content-path path/to/simple2017/content_with_label.png --out-path C:\Coding\Python\WCT_tf_final\output --passes 1
   ```

* 6 Apply Haze Removal to conclusion, online in http://tools.pfchai.com <br/>or with the script located in https://github.com/pfchai/Haze-Removal/blob/master/HazeRemovalWidthGuided.py
<br/>

<br/>
<br/>

<table>
<tr>
<td><img src="output/content_with_label_style_with_label_after_filter_maxpooling.jpg"/></td>
<td><img src="output/content_with_label_style_with_label_after_filter.png"/></td>
</tr>
</table>

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/PhotoWCT](https://github.com/svjack/PhotoWCT)
