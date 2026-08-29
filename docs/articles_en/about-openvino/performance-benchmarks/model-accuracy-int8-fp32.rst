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
     - 3.14%
     - 2.65%
     - 2.95%
     - 2.73%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.94%
     - -0.87%
     - -0.94%
     - -1.07%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.16%
     - -0.16%
     - -0.16%
     - -0.20%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.03%
     - 0.02%
     - -0.03%
     - 0.05%

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
     - 0.01%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.00%
     - -0.00%
     - -0.00%
     - -0.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.00%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.01%
     - 0.02%
     - 0.02%
   * - yolo_v11
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - -0.03%
     - -2.21%
     - -2.21%
     - 
   * - yolo_v26
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - 0.00%
     - 0.00%
     - 0.02%
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
     - 98.8%
     - 95.6%
     - 99.8%
     - 94.9%
   * - Gemma-3-4B-it
     - Data Default WWB
     - Similarity
     - 91.8%
     - 85.3%
     - 90.0
     - 86.0%
   * - GPT-OSS-20B
     - Data Default WWB
     - Similarity
     - 94.2%
     - 92.3%
     - 
     - 94.1%
   * - GPT-OSS-120B
     - Data Default WWB
     - Similarity
     - 98.1%
     - 94.2%
     - 
     - 
   * - Llama-2-7B-chat
     - Data Default WWB
     - Similarity
     - 98.7%
     - 94.4%
     - 99.9%
     - 93.4%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 98.7%
     - 91.9%
     - 99.9%
     - 94.3%
   * - MiniCPM-V-2.6
     - Data Default WWB
     - Similarity
     - 93.7%
     - 90.3%
     - 97.4%
     - 90.4%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 97.2%
     - 94.9%
     - 99.5%
     - 92.3%
   * - Qwen2.5-VL-7B
     - Data Default WWB
     - Similarity
     - 92.9%
     - 89.5%
     - 93.6%
     - 90.2%
   * - Qwen3-8B
     - Data Default WWB
     - Similarity
     - 99.2%
     - 92.7%
     - 99.9%
     - 90.8%
   * - Qwen3-30B-A3B
     - Data Default WWB
     - Similarity
     - 97.1%
     - 93.0%
     - 
     - 93.8%
   * - Qwen3.6-27B
     - Data Default WWB
     - Similarity
     - 98.2%
     - 93.4%
     - 
     - 
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 99.6%
     - 97.5%
     - 
     - 96.2%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 94.9%
     - 97.1%
     - 94.3%
     - 99.4%
   * - LTX-VIDEO
     - Data Default WWB
     - Similarity
     - 99.7%
     - 94.6%
     - 64.1%
     - 57.6%

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
