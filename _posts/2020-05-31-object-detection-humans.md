---
title: "Object Detection - Humans"
date: 2020-05-31
tags: [deep learning, computer vision, artificial intelligence]
header: 
    image: "/images/data_art.png"
excerpt: "Object Detection of Persons in an image using pretrained Darknet 53 YOLOv3 architecture"
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

    code {
        color: #f25278;
    }

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

Deep Learning has revolutionized the domain of Computer Vision. Over the years, research has proved successful to enable computers to mimic human perspective of the real world. This research has widened the scope of using Computer softwares for tasks which were otherwise tedious for humans thereby also reducing human error in such tasks. A good application of Computer Vision is to detect persons in a video or an image and monitor any suspicious activity used in highly secure rooms, for CCTV monitoring and for Sports Analysis. In case of Sports Analysis, player movements are traced to study their tactics and strategize for future games.

<b>Tools Used:</b>
> * <i>Python 3.6.9</i>
> * <i>Mxnet</i> 
> * <i>GluonCV</i>
> * <i>Matplotlib</i>

Object Detection is a subdomain of Computer Vision in which objects present in a frame or image are represented using a rectangle otherwise known as a bounding box. Each of these bounding boxes can be represented using one of the following two formats:

1. Pascal-VOC bounding box: <i>(x-top left, y-top left, x-bottom-right, y-bottom-right)</i>
> The (x-top-left, y-top-left) together give the absolute top-left point of the bounding box considering the top-left point of the image as the origin (0,0). The (x-bottom-right, y-bottom-right) give the absolute >bottom-right point of the image considering the top-left point of the image as the origin. 


2. COCO bounding box: <i>(x-top-left, y-top-left, width, height)</i>
> The x-top-left and y-top-left give the absolute top-left point of the bounding box considering the top-left point of the image as the origin (0,0). The width and height are then given relative to total width and height of image respectively. The COCO bounding box format is used for YOLO architectures. 


Researchers successfully developed three different algorithms which can be used to detect objects in an image. 
> * Regions with Convolutional Neural Networks (R-CNN)
> * Single Shot Detector (SSD)
> * You Look Only Once (YOLO)


Each of these algorithms has its advantages and drawbacks. The Faster R-CNN with Resnet 101 architecture pretrained on COCO dataset performs slower as compared to YOLOv3 with Darknet 53 architecture. However the MAP (Mean Average Precision) score, a metric used to compare Object Detection algorithms is higher for R-CNN than YOLOv3 for COCO test dataset. This implies that YOLOv3 architecture can process more frames per second with fairly lower accuracy whereas a Faster R-CNN architecture will process less frames per second but with higher accuracy.

This Project involves detecting humans in an image with bounding boxes. An application of this Project is to trace human movement in an environment from one frame to next used in Sports Analysis to study movement of players and determine new tactics to strategize for future games. Since the primary requirement for video analysis is to process higher frames per second (atleast 30), I chose the YOLOv3 with Darknet 53 architecture for this Project.

To implement the architecture, I used the Apache Mxnet API built on top of Pytorch. The Mxnet API has its own implementation of ndarray similar to numpy ndarray. The Mxnet Gluon CV library has a sophisticated Model Zoo with most popular Deep Learning architectures. Having used Tensorflow 1.x,Tensorflow 2.x and Pytorch for Deep Learning Projects, I can agree that Gluon CV has a more simpler and straightforward API to use pretrained models than the formers. In this Project, I used the YOLOv3 with Darknet 53 architecture pretrained on COCO dataset using one line `model = gluoncv.model_zoo.get_model('yolo3_darknet52_coco, pretrained=True)`.

<button type="button" id="coll_button" class="collapsible" onclick="toggleDivZoo()">Model zoo models</button>

<!--<div class="content" id="coll_content">
    <p> Hey </p>
</div>-->


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

<br />
To streamline the whole process from loading image to detecting humans and counting the number of humans in the image, I created a `PersonCounter` class. Any raw image requires preprocessing to be used with the model.The shortest dimension of the image is downsized to 416px and the other dimension is downsized proportionally. Also, the pixel values of the original image as 8 bit integers (0-255) are scaled to 0-1 and then normalized using <b>mean</b> of <i>0.485, 0.456, 0.406</i> and <b>standard deviation</b> of <i>0.229, 0.224, 0.225</i> accross the three channels (RGB channels). The `PersonCounter` class contains three methods `set_threshold`, `count` and  `_visualize`. The `set_threshold` method is used to set the minimum confidence score in order for the detected bounding box to be counted as a prediction. If the confidence score of the bounding box is less than threshold value, it is not counted as a prediction. Since the raw image used to detect humans with the pretrained model is preprocessed using multiple transformations, the `_visualize` method comes in handy to draw predicted bounding boxes on the untransformed original image. 
Finally, the `count` method runs the complete lifecycle of the Project from loading image, transforming the loaded image, detecting bounding boxes, visualizing bounding boxes and counting the number of people in the image. The code below shows the `PersonCounter` class and its methods. To keep the code readable and easy to debug, I used various helper functions such as `load_image`, `show_image`, `preprocess_image`, `detect`, `count_object` which can be found in the Github repo linked at the end of this document.

~~~~python
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
~~~~
<br />

---

<b>Demo</b>

As an FC Barcelona fan, I used multiple images of the dream team available on the Internet under Creative Commons License to detect humans within the image.

<img style="width:20px; height:20px" src="" alt="FC Barcelona" >
<img src="" alt="FC Barcelona" >

---

<!--<b>Code Repository</b>

Click <a href="https://github.com/kasim95/Object_Detection_Humans" target="_blank">here</a> to access the Github repository. 

---

-->
