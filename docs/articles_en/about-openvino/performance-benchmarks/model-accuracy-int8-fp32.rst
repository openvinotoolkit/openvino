Model Accuracy
==============

between OV-accuracy and the original framework accuracy for FP32, and the same for INT8, BF16,
and FP16 representations of a model on three platform architectures. The third table presents
the GenAI model accuracies as absolute accuracy values. Refer to notes below the table for more
information.

* A - Intel® Core™ i9-9000K (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Xeon 8580 (VNNI, AMX), INT8, BF16, FP32
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
     - 3.06%
     - 2.89%
     - 2.71%
     - 2.71%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - -0.84%
     - -0.59%
     - -0.59%
     - -0.55%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -0.10%
     - -0.04%
     - 0.07%
     - -0.01%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     -
     - -0.97%
     - -0.98%
     - -0.95%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.74%
     - 0.76%
     - 0.74%
     - 0.82%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.06%
     - -0.08%
     - -0.07%
     - -0.06%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - -2.94%
     - -0.28%
     - -0.28%
     - -0.26%
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
     - 0.02%
   * - efficientdet-d0
     - COCO2017_detection_91cl
     - coco_precision
     - 0.01%
     - 0.01%
     - 0.01%
     - 0.00%
     - 0.00%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - -0.01%
     - -0.01%
     - -0.01%
     - 0.05%
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
     - 0.01%
     - 0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.02%
     - -0.01%
     - 0.02%
   * - ssd-mobilenet-v1-coco
     - COCO2017_detection_80cl_bkgr
     - coco-precision
     - 0.04%
     - 0.01%
     - 0.04%
     - 0.08%
     - 0.01%
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
   * - chatGLM4
     - Wikiset
     - ppl
     -
     -
     -
     -
   * - Gemma-2-9B
     - Wikitext
     - ppl
     -
     - 1.57
     - 1.57
     -
   * - Llama-2-7b-chat
     - Wikiset
     - ppl
     -
     -
     - 1.59
     -
   * - Llama-3-8b
     - Wikiset
     - ppl
     - 1.45
     - 1.48
     - 1.45
     -
   * - Llama-3.2-3b-instruct
     - Wikiset
     - ppl
     - 1.60
     - 1.62
     - 1.17
     -
   * - Mistral-7b
     - Wikitext
     - ppl
     - 1.48
     - 1.49
     - 1.48
     -
   * - Phi3-mini-4k-instruct
     - Wikitext
     - ppl
     - 1.52
     - 1.55
     - 1.52
     - 1.56
   * - Qwen-2-7B
     - Wikitext
     - ppl
     - 1.52
     - 1.53
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
