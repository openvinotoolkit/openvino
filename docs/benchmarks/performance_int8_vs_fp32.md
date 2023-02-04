# Model Accuracy and Performance for INT8 and FP32 {#openvino_docs_performance_int8_vs_fp32}

The following table presents the absolute accuracy drop calculated as the accuracy difference between FP32 and INT8 representations of a model on two platforms

* A - Intel® Core™ i9-9000K (AVX2)
* B - Intel® Xeon® 6338, (VNNI)
* C - Intel® Flex-170

@sphinxdirective
.. list-table:: Model Accuracy
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A
     - B
     - C
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - 0.11%
     - 1.15%
     - 0.57%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.51%
     - 0.55%
     - 0.68%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.44%
     - 0.06%
     - 0.04%
   * - densenet-121
     - ImageNet2012
     - accuracy @ top1
     - 0.31%
     - 0.32%
     - 0.30%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - 0.88%	
     - 0.62%	
     - 0.50%
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_precision
     - 0.19%
     - 0.19%
     - 0.20%
   * - googlenet-v4
     - ImageNet2012_bkgr
     - accuracy @ top1
     - 0.07%
     - 0.09%
     - 0.26%
   * - mobilenet-ssd
     - VOC2007_detection
     - map
     - 0.47%
     - 0.14%
     - 0.48%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.50%
     - 0.18%
     - 0.20%
   * - resnet-18
     - ImageNet2012
     - accuracy @ top1
     - 0.27%
     - 0.24%
     - 0.29%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.13%
     - 0.12%
     - 0.13%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.08%
     - 0.09%
     - 0.06%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.33%
     - 0.33%	
     - 0.30%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - 0.01%
     - 0.07%
     - 0.12%
   * - yolo_v4
     - COCO2017_detection_80cl
     - map
     - 0.05%
     - 0.06%
     - 0.01%

@endsphinxdirective