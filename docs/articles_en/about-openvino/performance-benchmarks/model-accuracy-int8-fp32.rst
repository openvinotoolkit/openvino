Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures. The third table presents the GenAI model accuracies as absolute accuracy values. Please also refer to notes below
the table for more information.

* A - Intel® Core™ Ultra 9-185H (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Xeon 6972P (VNNI, AMX), INT8, BF16, FP32
* D - Intel® Arc-B60, INT8 and FP16


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
     - 2.60%
     - 2.70%
     - 3.00%
     - 2.60%
   * - Detectron-V2
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 
     - 
     - 
     - 
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.91%
     - -0.91%
     - -0.91%
     - -1.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.23%
     - -0.23%
     - -0.20%
     - -0.23%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.02%
     - 0.02%
   * - yolo_v11
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - 
     - 
     - 
     - 
.. list-table:: Model Accuracy for BF16, FP32 and FP16 (FP16: Arc only. BF16: Xeon® 6972P only)
   :header-rows: 1

   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, FP32
     - B, FP32
     - C, FP32
     - D, FP16
   * - bert-base-cased
     - SST-2_bert_cased_padded
     - spearman@cosine
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.00%
   * - Detectron-V2
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 
     - 
     - 
     - 
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.01%
     - -0.01%
     - -0.01%
     - -0.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.01%
     - 0.01%
     - 0.01%
     - 0.02%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.02%
     - 0.02%
   * - yolo_v11
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - 0.00%
     - 0.00%
     - 
     - 
.. list-table:: Model Accuracy for AMX-FP16, AMX-INT4, Arc-FP16 and Arc-INT4 (Arc™ B-series)
   :header-rows: 1
   
   * - OpenVINO™  Model name
     - dataset
     - Metric Name
     - A, AMX-FP16
     - B, AMX-INT4
     - C, Arc-FP16
     - D, Arc-INT4
   * - DeepSeek-R1-Distill-Llama-8B
     - Data Default WWB
     - Similarity
     - 97.2%
     - 94.1%
     - 92.0%
     - 93.3%
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - Data Default WWB
     - Similarity
     - 96.4%
     - 92.4%
     - 99.7%
     - 92.7%
   * - Gemma-3-4B-it
     - Data Default WWB
     - Similarity
     - 91.3%
     - 86.0%
     - 91.1%
     - 83.3%
   * - GPT-OSS-20B
     - Data Default WWB
     - Similarity
     - 94.3%
     - 90.8%
     - 
     - 91.2%
   * - Llama-2-7B-chat
     - Data Default WWB
     - Similarity
     - 99.0%
     - 93.2%
     - 96.2%
     - 93.2%
   * - Llama-3-8B
     - Data Default WWB
     - Similarity
     - 98.6%
     - 94.7%
     - 97.7%
     - 93.7%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 97.9%
     - 94.5%
     - 95.1%
     - 95.0%
   * - MiniCPM-V-2.6
     - Data Default WWB
     - Similarity
     - 90.7%
     - 88.4%
     - 95.3%
     - 95.3%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 96.0%
     - 92.5%
     - 93.7%
     - 91.5%
   * - Qwen2.5-VL-7B
     - Data Default WWB
     - Similarity
     - 91.1%
     - 90.2%
     - 91.4%
     - 89.9%
   * - Qwen3-8B
     - Data Default WWB
     - Similarity
     - 97.3%
     - 92.4%
     - 93.5%
     - 93.2%
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 99.0%
     - 96.1%
     - 
     - 
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 99.8%
     - 95.1%
     - 99.5%
     - 91.0%

Notes: For all accuracy metrics a "-", (minus sign), indicates an accuracy drop.
The Similarity metric is the distance from "perfect" and as such always positive. 
Similarity is cosine similarity - the dot product of two vectors divided by the product of their lengths.

.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see
   :doc:`F.A.Q. <./performance-benchmarks-faq>` and
   :doc:`Platforms, Configurations, Methodology <../performance-benchmarks>`.
   See :doc:`Legal Information <../additional-resources/terms-of-use>`.
