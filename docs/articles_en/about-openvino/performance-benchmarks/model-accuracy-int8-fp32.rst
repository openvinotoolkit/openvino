Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original framework accuracy for FP32, and the same for INT8, BF16,
and FP16 representations of a model on three platform architectures. The third table presents
the GenAI model accuracies as absolute accuracy values. Refer to notes below the table for more
information.

* A - Intel® Core™ i9-9000K (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Xeon 8480+ (VNNI, AMX), INT8, BF16, FP32
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
     - 3.33%
     - 3.22%
     - 3.05%
     - 2.88%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.12%
     - 0.03%
     - 0.03%
     - 0.28%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - 0.00%
     - -0.52%
     - -0.54%
     - -0.60%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 0.05%
     - 0.03%
     - 0.08%
     - -0.09%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     -
     - -0.87%
     - -0.88%
     - -0.88%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.17%
     - -0.18%
     - -0.18%
     - -0.16%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.03%
     - -0.02%
     - -0.03%
     - 0.02%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -2.74%
     - -0.11%
     - -0.13%
     - -0.12%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - -6.28%
     - 6.45%
     - 6.46%
     - 6.43%
   * - yolo_v5m
     - COCO2017_detection_80cl
     - map
     - -0.40%
     - -0.32%
     - -0.32%
     - -0.31%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - -0.01%
     - -0.04%
     - -0.07%
     - 0.05%

.. list-table:: Model Accuracy for BF16, FP32 and FP16 (FP16: Flex-170 only. BF16: Xeon(R) 8480+ only)
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
     - -0.01%
     - 0.01%
   * - bert-large-uncased-whole-word-masking-squad-0001
     - SQUAD_v1_1_bert_msl384_mql64_ds128_lowercase
     - F1
     - 0.04%
     - 0.04%
     - 0.06%
     - 0.06%
     - 0.04%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - 0.01%
     - -0.02%
     - 0.01%
     - 0.00%
     - -0.02%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -0.01%
     - -0.01%
     - -0.01%
     - -0.05%
     - 0.00%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.18%
     - 0.02%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.01%
     - -0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.00%
     - 0.00%
     - -0.02%
     - 0.04%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -0.08%
     - 0.01%
     - 0.01%
     - 0.08%
     - 0.01%
   * - unet-camvid-onnx-0001
     - CamVid_12cl
     - mean_iou @ mean
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.03%
     - -0.03%
   * - yolo_v5m
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.05%
     - 0.05%
     - 0.07%
     - 0.07%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.00%
     - 0.00%
     - 0.01%
     - 0.05%
     - 0.00%

.. list-table:: Model Accuracy for VNNI-FP16, VNNI-INT4, AMX-FP16 and MTL-INT4 (Core Ultra iGPU)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, VNNI-FP16
     - B, VNNI-INT4
     - C, FAMX-FP16
     - D, MTL-INT4
   * - chatGLM2-6b
     - Wikiset
     - ppl
     - 5.24
     - 6.03
     - 5.24
     - 6.03
   * - Falcon-7b-instruct
     - Wikitext
     - ppl
     - 1.65
     - 1.76
     - 1.65
     - 1.76
   * - Llama-2-7b-chat
     - Wikiset
     - ppl
     - 1.58
     - 1.59
     - 1.91
     - 1.59
   * - Llama-3-8b
     - Wikiset
     - ppl
     - 1.54
     - 1.56
     - 1.17
     - 1.57
   * - Mistral-7b
     - Wikitext
     - ppl
     - 1.48
     - 1.49
     - 1.39
     - 1.49
   * - Phi3-mini-4k-instruct
     - Wikitext
     - ppl
     - 1.52
     - 1.56
     - 1.52
     - 1.56

Notes: For all accuracy metrics a "-", (minus sign), indicates an accuracy drop.
For perplexity (ppl) the values do not indicate a deviation from a reference but are the actual measured
accuracy for the model.


.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see
   :doc:`F.A.Q. <./performance-benchmarks-faq>` and
   :doc:`Platforms, Configurations, Methodology <../performance-benchmarks>`.
   See :doc:`Legal Information <../additional-resources/terms-of-use>`.
