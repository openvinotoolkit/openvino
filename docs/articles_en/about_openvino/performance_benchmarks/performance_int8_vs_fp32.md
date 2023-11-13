# Model Accuracy {#openvino_docs_performance_int8_vs_fp32}


@sphinxdirective

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
     - -3.00%
     - -2.00%
     - 2.94%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - -0.04%
     - 0.03%
     - 0.06%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.00%
     - 0.23%
     - -0.13%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 
     - 0.97%
     - -0.97%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.20%
     - 0.12%
     - -0.19%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - 2.97%
     - 0.29%
     - -0.31%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.06%
     - 0.06%
     - -0.06%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 6.32%
     - -6.40%	
     - -0.63%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - -0.06%
     - -0.21%
     - -0.71%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - 0.73%
     - 0.21%
     - -0.78%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - -0.26%
     - -0.22%
     - 0.12%
   * - bloomz-560m
     - ROOTS corpus
     - ppl
     - 
     - 
     - 
   * - GPT-J-6B
     - Pile dataset
     - ppl
     - 
     - 4.11
     - 4.11
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     - 
     - 3.27
     - 3.27
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
     - -0.19%
     - 0.04%
     - 0.04%
   * - deeplabv3
     - VOC2012_segm
     - mean_iou
     - 0.49%
     - 0.00%
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
     - -0.02%
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
     - 0.01%
     - 0.06%
     - -0.06%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.02%
     - -6.45%	
     - 6.45%
   * - yolo_v3
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.01%
     - 0.01%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - -0.02%
     - 0.02%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
   * - bloomz-560m
     - ROOTS corpus
     - ppl
     - 
     - 22.89
     - 22.89
   * - GPT-J-6B
     - Pile dataset
     - ppl
     - 
     - 4.10
     - 4.10
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     - 
     - 2.91
     - 2.91
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - ppl
     - 
     - 
     -

Notes: For all accuracy metrics except perplexity a "-", (minus sign), indicates an accuracy drop. 
For perplexity the values do not indicate a deviation from a reference but are the actual measured accuracy for the model. 

@endsphinxdirective