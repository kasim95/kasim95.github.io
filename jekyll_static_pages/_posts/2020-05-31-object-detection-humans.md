---
title: "Object Detection in images"
date: 2020-05-31
tags: [deep learning, computer vision, artificial intelligence]
header: 
    overlay_color: "#ffffff"
    overlay_image: "/assets/images/bgs/pattern_1.png"
    overlay_filter: .4
    caption: "Photo Credit: [**Lukas**](http://simpledesktops.com/browse/desktops/2015/jun/16/pattern/)"
excerpt: "Recognizing humans in an image with bounding boxes using YOLOv3 architecture"
classes: wide
---


<script>
    function toggleDivZoo() {
        var div = document.getElementById("coll_content");
        if (div.style.display === "block") {
            div.style.display = "none";
        }
        else {
            div.style.display = "block";
        }
    }
</script>

<style>
    .collapsible {
        background-color: #eee;
        color: #444;
        cursor: pointer;
        padding: 5px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 15px;
        border-radius:5px;
    }

    .active {
        background-color: #ccc;
    }

    .collapsible:hover {
        background-color: #ccc;
    }

    .content {
        color: #000000;
        padding: 0 18px;
        display: none;
        overflow-y: auto;
        height: 400px;
        background-color: #f1f1f1;
    }

    /*code {
        color: #f25278;
    }
    */

    b {
        color: #f25278;
    }

    i {
        color: #f25278;
    }

    body {
        text-align: justify;
        font-size: 18px;
    }

</style>

Deep Learning has revolutionized the domain of Computer Vision. Over the years, research 
has proved successful to enable computers to mimic human perspective of the real world. 
This research has widened the scope of using Computer softwares for tasks which were 
otherwise tedious for humans as a result also reducing human error. A good 
application of Computer Vision is detecting humans in a video or an image to monitor 
suspicious activities or study human behavior. In case of Sports Analysis, player 
movements are traced to study their tactics and strategize for future games.

This Project involved recognizing humans in an image with boundind boxes using a 
pretrained YOLOv3 model based on Darknet 53 architecture.

<b>Tools Used:</b>
> * <i>Python 3</i>
> * <i>Apache MXNet</i> 
> * <i>GluonCV</i>
> * <i>Matplotlib</i>
> * <i>Jupyter Lab</i>

Object Detection is a subdomain of Computer Vision in which objects present in a frame 
or image are represented using a rectangle otherwise known as a bounding box. Each of 
these bounding boxes can be represented using one of the following two formats:

1. Pascal-VOC bounding box: <i>(x-top left, y-top left, x-bottom-right, y-bottom-right)</i>
> The *(x-top-left, y-top-left)* together gives the Cartesian coordinates of top-left point 
of the bounding box assuming the top-left point of the image as the origin (0,0). 
The *(x-bottom-right, y-bottom-right)* give the Cartesian coordinates of the bottom-right 
point of the image with the same assumptions.

2. COCO bounding box: <i>(x-top-left, y-top-left, width, height)</i>
> The *(x-top-left, y-top-left)* gives the Cartesian coordinates for top-left point of 
the bounding box considering the top-left point of the image as the origin (0,0). 
The *width* and *height* of the bounding boxare then given relative to total width 
and height of the image respectively. The COCO bounding box format is used for 
YOLO architectures. 

Researchers successfully developed three different algorithms for object detection as follows: 
> * Regions with Convolutional Neural Networks (R-CNN)
> * Single Shot Detector (SSD)
> * You Look Only Once (YOLO)

Each of these algorithms has its advantages and drawbacks. The Faster R-CNN with 
Resnet 101 architecture pretrained on COCO dataset performs slower as compared to 
YOLOv3 with Darknet 53 architecture. However the MAP (Mean Average Precision) score, 
a metric used to compare Object Detection algorithms; is higher for R-CNN than YOLOv3 
for COCO test dataset. This implies that YOLOv3 architecture can process more frames 
per second with fairly lower accuracy whereas a Faster R-CNN architecture will process 
less frames per second but with higher accuracy.

This Project involves detecting humans in an image with bounding boxes. An application 
of this Project is to trace human movement in an environment from one frame to next used 
in Sports Analysis to study movement of players and determine new tactics to strategize 
for future games. Since the primary requirement for video analysis is to process higher 
frames per second (atleast 30), I chose the YOLOv3 with Darknet 53 architecture for this
Project.

To implement the architecture, I used the Apache Mxnet API built on top of Pytorch. The 
Mxnet API has its own implementation of ndarray similar to numpy ndarray. The Mxnet Gluon 
CV library has a sophisticated Model Zoo with most popular Deep Learning architectures. 
Having used Tensorflow 1.x,Tensorflow 2.x and Pytorch for Deep Learning Projects, I can 
agree that Gluon CV has a more simpler and straightforward API to use pretrained models 
than the formers. In this Project, I used the YOLOv3 with Darknet 53 architecture 
pretrained on COCO dataset using the syntax below: 
`model = gluoncv.model_zoo.get_model('yolo3_darknet53_coco, pretrained=True)`.


<button 
    type="button" 
    id="coll_button" 
    class="collapsible" 
    onclick="toggleDivZoo()">GluonCV Model zoo pretrained architectures</button>
<div class="content" id="coll_content">
    <ul>
        <li>resnet18_v1</li>
        <li>resnet34_v1</li>
        <li>resnet50_v1</li>
        <li>resnet101_v1</li>
        <li>resnet152_v1</li>
        <li>resnet18_v2</li>
        <li>resnet34_v2</li>
        <li>resnet50_v2</li>
        <li>resnet101_v2</li>
        <li>resnet152_v2</li>
        <li>resnest50</li>
        <li>resnest101</li>
        <li>resnest200</li>
        <li>resnest269</li>
        <li>se_resnet18_v1</li>
        <li>se_resnet34_v1</li>
        <li>se_resnet50_v1</li>
        <li>se_resnet101_v1</li>
        <li>se_resnet152_v1</li>
        <li>se_resnet18_v2</li>
        <li>se_resnet34_v2</li>
        <li>se_resnet50_v2</li>
        <li>se_resnet101_v2</li>
        <li>se_resnet152_v2</li>
        <li>vgg11</li>
        <li>vgg13</li>
        <li>vgg16</li>
        <li>vgg19</li>
        <li>vgg11_bn</li>
        <li>vgg13_bn</li>
        <li>vgg16_bn</li>
        <li>vgg19_bn</li>
        <li>alexnet</li>
        <li>densenet121</li>
        <li>densenet161</li>
        <li>densenet169</li>
        <li>densenet201</li>
        <li>squeezenet1.0</li>
        <li>squeezenet1.1</li>
        <li>googlenet</li>
        <li>inceptionv3</li>
        <li>xception</li>
        <li>xception71</li>
        <li>mobilenet1.0</li>
        <li>mobilenet0.75</li>
        <li>mobilenet0.5</li>
        <li>mobilenet0.25</li>
        <li>mobilenetv2_1.0</li>
        <li>mobilenetv2_0.75</li>
        <li>mobilenetv2_0.5</li>
        <li>mobilenetv2_0.25</li>
        <li>mobilenetv3_large</li>
        <li>mobilenetv3_small</li>
        <li>mobile_pose_resnet18_v1b</li>
        <li>mobile_pose_resnet50_v1b</li>
        <li>mobile_pose_mobilenet1.0</li>
        <li>mobile_pose_mobilenetv2_1.0</li>
        <li>mobile_pose_mobilenetv3_large</li>
        <li>mobile_pose_mobilenetv3_small</li>
        <li>ssd_300_vgg16_atrous_voc</li>
        <li>ssd_300_vgg16_atrous_coco</li>
        <li>ssd_300_vgg16_atrous_custom</li>
        <li>ssd_512_vgg16_atrous_voc</li>
        <li>ssd_512_vgg16_atrous_coco</li>
        <li>ssd_512_vgg16_atrous_custom</li>
        <li>ssd_512_resnet18_v1_voc</li>
        <li>ssd_512_resnet18_v1_coco</li>
        <li>ssd_512_resnet50_v1_voc</li>
        <li>ssd_512_resnet50_v1_coco</li>
        <li>ssd_512_resnet50_v1_custom</li>
        <li>ssd_512_resnet101_v2_voc</li>
        <li>ssd_512_resnet152_v2_voc</li>
        <li>ssd_512_mobilenet1.0_voc</li>
        <li>ssd_512_mobilenet1.0_coco</li>
        <li>ssd_512_mobilenet1.0_custom</li>
        <li>ssd_300_mobilenet0.25_voc</li>
        <li>ssd_300_mobilenet0.25_coco</li>
        <li>ssd_300_mobilenet0.25_custom</li>
        <li>faster_rcnn_resnet50_v1b_voc</li>
        <li>mask_rcnn_resnet18_v1b_coco</li>
        <li>faster_rcnn_resnet50_v1b_coco</li>
        <li>faster_rcnn_fpn_resnet50_v1b_coco</li>
        <li>faster_rcnn_fpn_syncbn_resnet50_v1b_coco</li>
        <li>faster_rcnn_fpn_syncbn_resnest50_coco</li>
        <li>faster_rcnn_resnet50_v1b_custom</li>
        <li>faster_rcnn_resnet101_v1d_voc</li>
        <li>faster_rcnn_resnet101_v1d_coco</li>
        <li>faster_rcnn_fpn_resnet101_v1d_coco</li>
        <li>faster_rcnn_fpn_syncbn_resnet101_v1d_coco</li>
        <li>faster_rcnn_fpn_syncbn_resnest101_coco</li>
        <li>faster_rcnn_resnet101_v1d_custom</li>
        <li>faster_rcnn_fpn_syncbn_resnest269_coco</li>
        <li>custom_faster_rcnn_fpn</li>
        <li>mask_rcnn_resnet50_v1b_coco</li>
        <li>mask_rcnn_fpn_resnet50_v1b_coco</li>
        <li>mask_rcnn_resnet101_v1d_coco</li>
        <li>mask_rcnn_fpn_resnet101_v1d_coco</li>
        <li>mask_rcnn_fpn_resnet18_v1b_coco</li>
        <li>mask_rcnn_fpn_syncbn_resnet18_v1b_coco</li>
        <li>mask_rcnn_fpn_syncbn_mobilenet1_0_coco</li>
        <li>custom_mask_rcnn_fpn</li>
        <li>cifar_resnet20_v1</li>
        <li>cifar_resnet56_v1</li>
        <li>cifar_resnet110_v1</li>
        <li>cifar_resnet20_v2</li>
        <li>cifar_resnet56_v2</li>
        <li>cifar_resnet110_v2</li>
        <li>cifar_wideresnet16_10</li>
        <li>cifar_wideresnet28_10</li>
        <li>cifar_wideresnet40_8</li>
        <li>cifar_resnext29_32x4d</li>
        <li>cifar_resnext29_16x64d</li>
        <li>fcn_resnet50_voc</li>
        <li>fcn_resnet101_coco</li>
        <li>fcn_resnet101_voc</li>
        <li>fcn_resnet50_ade</li>
        <li>fcn_resnet101_ade</li>
        <li>psp_resnet101_coco</li>
        <li>psp_resnet101_voc</li>
        <li>psp_resnet50_ade</li>
        <li>psp_resnet101_ade</li>
        <li>psp_resnet101_citys</li>
        <li>deeplab_resnet101_coco</li>
        <li>deeplab_resnet101_voc</li>
        <li>deeplab_resnet152_coco</li>
        <li>deeplab_resnet152_voc</li>
        <li>deeplab_resnet50_ade</li>
        <li>deeplab_resnet101_ade</li>
        <li>deeplab_resnest50_ade</li>
        <li>deeplab_resnest101_ade</li>
        <li>deeplab_resnest200_ade</li>
        <li>deeplab_resnest269_ade</li>
        <li>deeplab_resnet50_citys</li>
        <li>deeplab_resnet101_citys</li>
        <li>deeplab_v3b_plus_wideresnet_citys</li>
        <li>icnet_resnet50_citys</li>
        <li>icnet_resnet50_mhpv1</li>
        <li>resnet18_v1b</li>
        <li>resnet34_v1b</li>
        <li>resnet50_v1b</li>
        <li>resnet50_v1b_gn</li>
        <li>resnet101_v1b_gn</li>
        <li>resnet101_v1b</li>
        <li>resnet152_v1b</li>
        <li>resnet50_v1c</li>
        <li>resnet101_v1c</li>
        <li>resnet152_v1c</li>
        <li>resnet50_v1d</li>
        <li>resnet101_v1d</li>
        <li>resnet152_v1d</li>
        <li>resnet50_v1e</li>
        <li>resnet101_v1e</li>
        <li>resnet152_v1e</li>
        <li>resnet50_v1s</li>
        <li>resnet101_v1s</li>
        <li>resnet152_v1s</li>
        <li>resnext50_32x4d</li>
        <li>resnext101_32x4d</li>
        <li>resnext101_64x4d</li>
        <li>resnext101b_64x4d</li>
        <li>se_resnext50_32x4d</li>
        <li>se_resnext101_32x4d</li>
        <li>se_resnext101_64x4d</li>
        <li>se_resnext101e_64x4d</li>
        <li>senet_154</li>
        <li>senet_154e</li>
        <li>darknet53</li>
        <li>yolo3_darknet53_coco</li>
        <li>yolo3_darknet53_voc</li>
        <li>yolo3_darknet53_custom</li>
        <li>yolo3_mobilenet1.0_coco</li>
        <li>yolo3_mobilenet1.0_voc</li>
        <li>yolo3_mobilenet1.0_custom</li>
        <li>yolo3_mobilenet0.25_coco</li>
        <li>yolo3_mobilenet0.25_voc</li>
        <li>yolo3_mobilenet0.25_custom</li>
        <li>nasnet_4_1056</li>
        <li>nasnet_5_1538</li>
        <li>nasnet_7_1920</li>
        <li>nasnet_6_4032</li>
        <li>simple_pose_resnet18_v1b</li>
        <li>simple_pose_resnet50_v1b</li>
        <li>simple_pose_resnet101_v1b</li>
        <li>simple_pose_resnet152_v1b</li>
        <li>simple_pose_resnet50_v1d</li>
        <li>simple_pose_resnet101_v1d</li>
        <li>simple_pose_resnet152_v1d</li>
        <li>residualattentionnet56</li>
        <li>residualattentionnet92</li>
        <li>residualattentionnet128</li>
        <li>residualattentionnet164</li>
        <li>residualattentionnet200</li>
        <li>residualattentionnet236</li>
        <li>residualattentionnet452</li>
        <li>cifar_residualattentionnet56</li>
        <li>cifar_residualattentionnet92</li>
        <li>cifar_residualattentionnet452</li>
        <li>resnet18_v1b_0.89</li>
        <li>resnet50_v1d_0.86</li>
        <li>resnet50_v1d_0.48</li>
        <li>resnet50_v1d_0.37</li>
        <li>resnet50_v1d_0.11</li>
        <li>resnet101_v1d_0.76</li>
        <li>resnet101_v1d_0.73</li>
        <li>mobilenet1.0_int8</li>
        <li>resnet50_v1_int8</li>
        <li>ssd_300_vgg16_atrous_voc_int8</li>
        <li>ssd_512_mobilenet1.0_voc_int8</li>
        <li>ssd_512_resnet50_v1_voc_int8</li>
        <li>ssd_512_vgg16_atrous_voc_int8</li>
        <li>alpha_pose_resnet101_v1b_coco</li>
        <li>vgg16_ucf101</li>
        <li>vgg16_hmdb51</li>
        <li>vgg16_kinetics400</li>
        <li>vgg16_sthsthv2</li>
        <li>inceptionv1_ucf101</li>
        <li>inceptionv1_hmdb51</li>
        <li>inceptionv1_kinetics400</li>
        <li>inceptionv1_sthsthv2</li>
        <li>inceptionv3_ucf101</li>
        <li>inceptionv3_hmdb51</li>
        <li>inceptionv3_kinetics400</li>
        <li>inceptionv3_sthsthv2</li>
        <li>c3d_kinetics400</li>
        <li>p3d_resnet50_kinetics400</li>
        <li>p3d_resnet101_kinetics400</li>
        <li>r2plus1d_resnet18_kinetics400</li>
        <li>r2plus1d_resnet34_kinetics400</li>
        <li>r2plus1d_resnet50_kinetics400</li>
        <li>r2plus1d_resnet101_kinetics400</li>
        <li>r2plus1d_resnet152_kinetics400</li>
        <li>i3d_resnet50_v1_ucf101</li>
        <li>i3d_resnet50_v1_hmdb51</li>
        <li>i3d_resnet50_v1_kinetics400</li>
        <li>i3d_resnet50_v1_sthsthv2</li>
        <li>i3d_resnet50_v1_custom</li>
        <li>i3d_resnet101_v1_kinetics400</li>
        <li>i3d_inceptionv1_kinetics400</li>
        <li>i3d_inceptionv3_kinetics400</li>
        <li>i3d_nl5_resnet50_v1_kinetics400</li>
        <li>i3d_nl10_resnet50_v1_kinetics400</li>
        <li>i3d_nl5_resnet101_v1_kinetics400</li>
        <li>i3d_nl10_resnet101_v1_kinetics400</li>
        <li>slowfast_4x16_resnet50_kinetics400</li>
        <li>slowfast_4x16_resnet50_custom</li>
        <li>slowfast_8x8_resnet50_kinetics400</li>
        <li>slowfast_4x16_resnet101_kinetics400</li>
        <li>slowfast_8x8_resnet101_kinetics400</li>
        <li>slowfast_16x8_resnet101_kinetics400</li>
        <li>slowfast_16x8_resnet101_50_50_kinetics400</li>
        <li>resnet18_v1b_kinetics400</li>
        <li>resnet34_v1b_kinetics400</li>
        <li>resnet50_v1b_kinetics400</li>
        <li>resnet101_v1b_kinetics400</li>
        <li>resnet152_v1b_kinetics400</li>
        <li>resnet18_v1b_sthsthv2</li>
        <li>resnet34_v1b_sthsthv2</li>
        <li>resnet50_v1b_sthsthv2</li>
        <li>resnet101_v1b_sthsthv2</li>
        <li>resnet152_v1b_sthsthv2</li>
        <li>resnet50_v1b_ucf101</li>
        <li>resnet50_v1b_hmdb51</li>
        <li>resnet50_v1b_custom</li>
        <li>fcn_resnet101_voc_int8</li>
        <li>fcn_resnet101_coco_int8</li>
        <li>psp_resnet101_voc_int8</li>
        <li>psp_resnet101_coco_int8</li>
        <li>deeplab_resnet101_voc_int8</li>
        <li>deeplab_resnet101_coco_int8</li>
        <li>center_net_resnet18_v1b_voc</li>
        <li>center_net_resnet18_v1b_dcnv2_voc</li>
        <li>center_net_resnet18_v1b_coco</li>
        <li>center_net_resnet18_v1b_dcnv2_coco</li>
        <li>center_net_resnet50_v1b_voc</li>
        <li>center_net_resnet50_v1b_dcnv2_voc</li>
        <li>center_net_resnet50_v1b_coco</li>
        <li>center_net_resnet50_v1b_dcnv2_coco</li>
        <li>center_net_resnet101_v1b_voc</li>
        <li>center_net_resnet101_v1b_dcnv2_voc</li>
        <li>center_net_resnet101_v1b_coco</li>
        <li>center_net_resnet101_v1b_dcnv2_coco</li>
        <li>center_net_dla34_voc</li>
        <li>center_net_dla34_dcnv2_voc</li>
        <li>center_net_dla34_coco</li>
        <li>center_net_dla34_dcnv2_coco</li>
        <li>dla34</li>
        <li>simple_pose_resnet18_v1b_int8</li>
        <li>simple_pose_resnet50_v1b_int8</li>
        <li>simple_pose_resnet50_v1d_int8</li>
        <li>simple_pose_resnet101_v1b_int8</li>
        <li>simple_pose_resnet101_v1d_int8</li>
        <li>vgg16_ucf101_int8</li>
        <li>inceptionv3_ucf101_int8</li>
        <li>resnet18_v1b_kinetics400_int8</li>
        <li>resnet50_v1b_kinetics400_int8</li>
        <li>inceptionv3_kinetics400_int8</li>
        <li>hrnet_w18_small_v1_c</li>
        <li>hrnet_w18_small_v2_c</li>
        <li>hrnet_w30_c</li>
        <li>hrnet_w32_c</li>
        <li>hrnet_w40_c</li>
        <li>hrnet_w44_c</li>
        <li>hrnet_w48_c</li>
        <li>hrnet_w18_small_v1_s</li>
        <li>hrnet_w18_small_v2_s</li>
        <li>hrnet_w48_s</li>
        <li>siamrpn_alexnet_v2_otb15</li>
    </ul>
</div>


---
<b>Code Walkthrough</b>

```python
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv import model_zoo, data, utils
import os
import matplotlib.pyplot as plt
from pathlib import Path
```


```python
cwd = Path()
pathImages = Path(cwd, 'images')
pathModels = Path(cwd, 'models')
```


```python
model_name = 'yolo3_darknet53_coco'
model = gcv.model_zoo.get_model(model_name, pretrained=True, root=pathModels)
```

### Helper Functions


```python
# read image as nd array
def load_image(path):
    return mx.nd.array(mx.image.imread(path))

# display nd array image
def show_image(array):
    plt.imshow(array)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.show()

# preprocess image using normalization and resizing to predict objects for yolov3 model
def preprocess_image(array):
     return gcv.data.transforms.presets.yolo.transform_test(array)

# detect objects within image using model
def detect(_model, _data):
    class_ids, scores, bounding_boxes = _model(_data)
    return class_ids, scores, bounding_boxes

# draw and display bounding boxes for detected objects on image
def draw_bbs(unnorm_array, bounding_boxes, scores, class_ids, all_class_names):
    ax = utils.viz.plot_bbox(unnorm_array, bounding_boxes, scores, class_ids, class_names=model.classes)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.show()

# count number of objects detected in image for an object_label
def count_object(network, class_ids, scores, bounding_boxes, object_label, threshold=0.75):
    target_idx = network.classes.index(object_label)
    num_objects = 0
    for i in range(len(class_ids[0])):
        if class_ids[0][i].asscalar() == target_idx and scores[0][i].asscalar() >= threshold:
            num_objects += 1
    return num_objects
    
```

### Load and display raw image

```python
image = load_image(Path(pathImages, '02.jpg'))
show_image(image.asnumpy())
```


![png](/assets/images/posts/object_detection_humans/output_6_0.png)


### Preprocess image

```python
norm_image, unnorm_image = preprocess_image(image)
show_image(unnorm_image)
```


![png](/assets/images/posts/object_detection_humans/output_7_0.png)


### Detect and draw bounding boxes on objects

```python
# Detect persons
class_ids, scores, bounding_boxes = detect(model, norm_image)
#
draw_bbs(unnorm_array = unnorm_image, 
         bounding_boxes=bounding_boxes[0], 
         scores=scores[0], 
         class_ids=class_ids[0], 
         all_class_names=model.classes
        )
```


![png](/assets/images/posts/object_detection_humans/output_8_0.png)


To streamline the process of loading image, preprocessing, detecting inference and 
counting the number of bounding boxes in the image, a `PersonCounter` class is used.
Any raw image requires preprocessing to be used with the model. The shortest dimension 
of the image is downsized to 416px and the other dimension is downsized proportionally. 
Also, the pixel values of the original image as 8 bit integers (0-255) are scaled to 0-1 
and then normalized using <b>mean</b> of <i>0.485, 0.456, 0.406</i> and <b>standard 
deviation</b> of <i>0.229, 0.224, 0.225</i> accross the RGB channels. 
The `PersonCounter` class contains three methods `set_threshold`, `count` 
and  `_visualize`. The `set_threshold` method is used to set the minimum confidence 
score in order for the detected bounding box to be counted as a prediction.
Since the image is transformed before finding the inference on the model,
the `_visualize` method comes in handy 
to draw predicted bounding boxes on the raw untransformed image. 
Finally, the `count` method is responsible for loading image, preprocessing it, 
detecting objects & visualizing them using bounding 
boxes, and finally counting the number of humans in the image. The code below shows 
the `PersonCounter` class and its methods.

```python
class PersonCounter():
    def __init__(self, threshold):
        self._network = gcv.model_zoo.get_model(model_name, 
                                                pretrained=True, 
                                                root=pathModels
                                               )
        self._threshold = threshold

    def set_threshold(self, threshold):
        self._threshold = threshold
        
    def count(self, filepath, visualize=False):
        # Load and Preprocess image
        image = load_image(filepath)
        if visualize:
            show_image(image.asnumpy())
        
        norm_image, unnorm_image = preprocess_image(image)
        
        # Detect persons
        class_ids, scores, bounding_boxes = detect(self._network, norm_image)
        #
        
        if visualize:
            self._visualize(unnorm_image, class_ids, scores, bounding_boxes)
        
        # Count no of persons
        num_people = count_object(
            network=self._network, 
            class_ids=class_ids,
            scores=scores,
            bounding_boxes=bounding_boxes,
            object_label="person",
            threshold=self._threshold)
        
        if num_people == 1:
            print('{} person detected in {} with minimum {} % confidence.'.format(num_people, filepath, self._threshold * 100)) 
        else:
            print('{} people detected in {} with minimum {} % confidence.'.format(num_people, filepath, self._threshold * 100))
        return num_people
    
    def _visualize(self, unnorm_image, class_ids, scores, bounding_boxes):
        draw_bbs(unnorm_array = unnorm_image, 
                 bounding_boxes=bounding_boxes[0], 
                 scores=scores[0], 
                 class_ids=class_ids[0], 
                 all_class_names=self._network.classes
                )
```


```python
counter = PersonCounter(threshold=0.6)

images = ['01.jpeg', '02.jpg', '03.jpg', '04.jpg']
for img in images:
    print('Image name', img, sep=":")
    counter.count(filepath=Path(pathImages, img), visualize=True)
    print('*'*50+'\n\n')
```

    Image name:01.jpeg



![png](/assets/images/posts/object_detection_humans/output_11_1.png)



![png](/assets/images/posts/object_detection_humans/output_11_2.png)


    4 people detected in images\01.jpeg with minimum 60.0 % confidence.
    **************************************************
    
    
    Image name:02.jpg



![png](/assets/images/posts/object_detection_humans/output_11_4.png)



![png](/assets/images/posts/object_detection_humans/output_11_5.png)


    9 people detected in images\02.jpg with minimum 60.0 % confidence.
    **************************************************
    
    
    Image name:03.jpg



![png](/assets/images/posts/object_detection_humans/output_11_7.png)



![png](/assets/images/posts/object_detection_humans/output_11_8.png)


    13 people detected in images\03.jpg with minimum 60.0 % confidence.
    **************************************************
    
    
    Image name:04.jpg



![png](/assets/images/posts/object_detection_humans/output_11_10.png)



![png](/assets/images/posts/object_detection_humans/output_11_11.png)


    3 people detected in images\04.jpg with minimum 60.0 % confidence.
    **************************************************
    
<!--
---
<b>Code Repository</b>

Click <a href="https://github.com/kasim95/Object_Detection_Humans" target="_blank">here</a> to access the Github repository. 
-->

