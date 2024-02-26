.. {#openvino_docs_performance_int8_vs_fp32}

Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures. Please also refer to notes below
The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures. Please also refer to notes below
the table for more information.

* A - Intel® Core™ i9-9000K (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel(R) Xeon 8490H (VNNI, AMX), INT8, BF16, FP32
* D - Intel® Flex-170, INT8 and FP16

.. list-table:: Model Accuracy for INT8
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, INT8
     - B, INT8
     - C, INT8
     - D, INT8
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - spearman@cosine
     - 3.17%
     - 2.68%
     - 3.00%
     - 2.73%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.07%
     - -0.03%
     - 0.13%
     - 0.11%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.84%
     - -0.59%
     - -0.62%
     - -0.63%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 0.03%
     - 0.08%
     - 0.11%
     - 0.07%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - %
     - -0.97%
     - -0.97%
     - -0.95%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.20%
     - -0.19%
     - -0.13%
     - -0.15%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.03%
     - -0.06%
     - -0.01%
     - 0.04%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -2.97%
     - -0.29%
     - -0.31%
     - -0.26%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - -6.32%
     - 6.40%
     - 6.41%
     - 6.40%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - %
     - -0.23%
     - -0.24%
     - -0.66%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - -0.02%
     - -0.03%
     - -0.06%
     - -0.06%
   * - chatGLM2-6b
     - lambada openai
     - ppl
     -
     - 17.38
     - 17.41
     - 17.17
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     -
     - 3.24
     - 3.24
     - 3.25
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - CLIP
     -
     -
     -
     -
   * - Mistral-7b
     - proprietary Mistral.ai
     - ppl
     -
     - 3.29
     - 3.47
     - 3.49

.. list-table:: Model Accuracy for BF16, FP32 and FP16 (FP16: Flex-170 only. BF16: Xeon(R) 8490H only)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, FP32
     - B, FP32
     - C, FP32
     - C, BF16
     - D, FP16
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - spearman@cosine
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.09%
     - 0.00%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.04%
     - 0.04%
     - 0.04%
     - 0.06%
     - 0.04%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.02%
     - -0.02%
     - -0.02%
     - -0.02%
     - -0.03%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -0.01%
     - -0.01%
     - %
     - -0.18%
     - 0.02%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.04%
     - 0.02%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.02%
     - 0.02%
     - 0.00%
     - 0.01%
     - 0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.02%
     - 0.02%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - 0.01%
     - 0.01%
     - 0.01%
     - 0.05%
     - -0.03%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.03%
     - -0.03%
   * - yolo_v3_tiny
     - COCO2017_detection_80cl
     - map
     - %
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.02%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.05%
     - -0.03%
   * - chatGLM2-6b
     - lambada openai
     - ppl
     -
     - 17.48
     - 17.56
     -
     - 17.49
   * - Llama-2-7b-chat
     - Wiki, StackExch, Crawl
     - ppl
     -
     - 3.26
     - 3.26
     -
     -
   * - Stable-Diffusion-V2-1
     - LIAON-5B
     - CLIP
     -
     -
     -
     -
     - 22.48
   * - Mistral-7b
     - proprietary Mistral.ai
     - ppl
     -
     - 3.19
     - 3.18
     -
     -

Notes: For all accuracy metrics except perplexity a "-", (minus sign), indicates an accuracy drop.
For perplexity (ppl) the values do not indicate a deviation from a reference but are the actual measured
accuracy for the model.
