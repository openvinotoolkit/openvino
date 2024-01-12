.. {#openvino_docs_performance_int8_vs_fp32}

Model Accuracy
==============



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
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - -0.76%
     - 2.42%
     - 2.72%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.07%
     - -0.03%
     - 0.11%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.49%
     - 0.23%
     - -0.16%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.84%
     - -0.59%
     - -0.63%
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -0.19%
     - -0.19%
     - -0.04%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 
     - -0.97%
     - -0.95%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.09%
     - -0.12%
     - -0.19%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -2.97%
     - -0.29%
     - -0.26%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.03%
     - -0.06%
     - 0.04%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - -6.32%
     - 6.40%	
     - 6.40%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - -0.13%
     - -0.26%
     - -0.44%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - -0.11%
     - -0.13%
     - -0.15%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.27%
     - 0.23%
     - 0.17%
   * - chatGLM2-6b
     - lambada openai
     - ppl
     - 
     - 17.595
     - 
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     - 
     - 3.268
     - 
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - ppl
     - 
     - 
     - 

.. list-table:: Model Accuracy for FP32 and FP16 (FP16: Flex-170 only)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, FP32
     - B, FP32
     - C, FP16
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - accuracy
     - 0.00%
     - 0.00%
     - 0.00%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.04%
     - 0.04%
     - 0.04%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.00%
     - 0.00%
     - 0.00%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.02%
     - -0.02%
     - -0.02%
   * - faster_rcnn_resnet50_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 0.00%
     - 
     - 0.00%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - 0.01%
     - 0.01%
     - 0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.00%
     - 0.00%
     - 0.00%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.00%
     - 0.00%	
     - 0.00%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - -0.04%
     - -0.04%
     - 0.02%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
   * - chatGLM2-6b
     - lambada-openai
     - ppl
     - 
     - 17.488
     - 
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     - 
     - 3.262
     - 
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - ppl
     - 
     - 
     -

Notes: For all accuracy metrics except perplexity a "-", (minus sign), indicates an accuracy drop. 
For perplexity the values do not indicate a deviation from a reference but are the actual measured accuracy for the model. 

