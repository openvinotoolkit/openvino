# OpenVINO Accuracy {#openvino_docs_performance_int8_vs_fp32}


@sphinxdirective

.. meta::
   :description: Learn about the differences in absolute accuracy drop for INT8, 
                 FP32 and FP16 representations of models inferred with OpenVINO 
                 on three different platforms.

The following two tables present the absolute accuracy drop calculated as the accuracy difference 
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8 and FP16 
representations of a model on three platform architectures. Please also refer to notes below table 
for more information.

* A - Intel® Core™ i9-9000K (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Flex-170, INT8 and FP16


.. list-table:: Model Accuracy for INT8
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, INT8
     - B, INT8
     - C, INT8
   * - GPT-2
     - WikiText_2_raw_gpt2
     - perplexity
     - n/a
     - n/a
     - n/a
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - 1.15%
     - 1.51%
     - -0.85%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.05%
     - 0.11%
     - 0.10%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - -0.46%
     - -0.23%
     - -0.18%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.87%	
     - -0.56%	
     - n/a
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_precision
     - -0.24%
     - -0.24%
     - 0.00%
   * - inception-v4
     - ImageNet2012_bkgr
     - accuracy @ top1
     - -0.06%
     - -0.08%
     - -0.04%
   * - mobilenet-ssd
     - VOC2007_detection
     - map
     - -0.49%
     - -0.50%
     - -0.47%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.70%
     - -1.11%
     - -1.05%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.13%
     - -0.11%
     - -0.14%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.02%
     - -0.03%
     - 0.04%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - n/a
     - 6.40%	
     - -0.30%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - -0.14%
     - -0.01%
     - -0.19%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - -0.11%
     - -0.13%
     - -0.17%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - n/a
     - n/a
     - n/a

.. list-table:: Model Accuracy for FP32 and FP16 (Flex-170 only)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, FP32
     - B, FP32
     - C, FP16
   * - GPT-2
     - WikiText_2_raw_gpt2
     - perplexity
     - -9.12%
     - -9.12%
     - -9.12%
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - 0.00%
     - 0.00%
     - 0.01%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.04%
     - 0.04%
     - 0.05%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.00%
     - 0.00%
     - 0.01%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.01%	
     - 0.02%	
     - 0.02%
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_precision
     - 0.00%
     - -0.01%
     - 0.03%
   * - inception-v4
     - ImageNet2012_bkgr
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.01%
   * - mobilenet-ssd
     - VOC2007_detection
     - map
     - 0.00%
     - 0.00%
     - 0.02%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.08%
     - -0.08%
     - 0.06%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.00%
     - 0.00%
     - 0.02%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - -0.02%
     - -0.02%	
     - 0.05%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - 0.02%
     - 0.02%
     - 0.03%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - -0.04%
     - -0.04%
     - 0.03%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.03%

.. note:: 
  
   For all accuracy metrics except perplexity a "-" (minus sign) indicates an accuracy drop.
   For perplexity a "-" indicates improved accuracy. 

@endsphinxdirective