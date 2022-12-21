# Model Accuracy and Performance for INT8 and FP32 {#openvino_docs_performance_int8_vs_fp32}

The following table presents the absolute accuracy drop calculated as the accuracy difference between FP32 and INT8 representations of a model on two platforms

* A - Intel® Core™ i9-9000K (AVX2)
* B - Intel® Xeon® 6338, (VNNI)

@sphinxdirective
.. list-table:: Model Accuracy
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A
     - B
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - 0.11%
     - 1.15%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.51%
     - 
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.44%
     - 0.06%
   * - densenet-121
     - ImageNet2012
     - accuracy@top1
     - 0.31%
     - 0.32%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - 0.88%	
     - 0.62%	
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_precision
     - 0.19%
     - 0.19%
   * - googlenet-v4
     - ImageNet2012_bkgr
     - accuracy@top1
     - 0.07%
     - 0.09%
   * - mobilenet-ssd
     - VOC2007_detection
     - map
     - 0.47%
     - 0.14%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy@top1
     - 0.50%
     - 0.18%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy@top1
     - 0.50%
     - 0.18%
   * - resnet-18
     - ImageNet2012
     - accuracy@top1
     - 0.27%
     - 0.24%
   * - resnet-50
     - ImageNet2012
     - accuracy@top1
     - 0.13%
     - 0.12%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.08%
     - 0.09%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou@mean
     - 0.33%
     - 0.33%	
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - 0.01%
     - 0.07%
   * - yolo_v4
     - COCO2017_detection_80cl
     - map
     - 0.05%
     - 0.06%

@endsphinxdirective