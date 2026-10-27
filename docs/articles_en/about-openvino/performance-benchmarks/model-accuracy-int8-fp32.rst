Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures (percent point). The third table presents the GenAI model accuracies as absolute accuracy values. Please also refer to notes below
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
     - 2.57%
     - 2.65%
     - 2.95%
     - 2.57%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -0.91%
     - -0.93%
     - -0.98%
     - -1.07%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.12%
     - -0.12%
     - -0.15%
     - -0.15%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.00%
     - 0.01%
     - 0.00%
     - 0.06%
   * - yolov26n
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.53%
     - -0.50%
     - -0.47%
     - -0.48%

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
     - 2.95%
     - 0.00%
     - 0.00%
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - -0.91%
     - 0.00%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - -0.15%
     - 0.00%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.04%
     - 0.02%
   * - yolo_v11
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - -0.03%
     - -2.21%
     - -2.21%
     - -2.21%
   * - yolo_v26
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - 0.00%
     - 0.00%
     - -0.47%
     - 0.00%
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
     - 97.9%
     - 91.5%
     - 98.2%
     - 92.8%
   * - Gemma3n-e4b-it
     - Data Default WWB
     - Similarity
     - %
     - %
     - %
     - %
   * - Gemma-4-e2b-it
     - Data Default WWB
     - Similarity
     - 98.8%
     - 90.9%
     - 98.6%
     - %
   * - Gemma-4-26b-a4b
     - Data Default WWB
     - Similarity
     - 97.4%
     - %
     - 98.8%
     - %
   * - GPT-OSS-20B
     - Data Default WWB
     - Similarity
     - 97.6%
     - 94.2%
     - %
     - 92.8%
   * - Kokoro-82m
     - Data Default WWB
     - Similarity
     - %
     - %
     - %
     - %
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 98.8%
     - 97.7%
     - 98.4%
     - 91.2%
   * - MiniCPM-V-2.6
     - Data Default WWB
     - Similarity
     - 94.2%
     - 94.9%
     - 94.9%
     - 91.1%
   * - Mistral-7B-instruct
     - Data Default WWB
     - Similarity
     - 98.5%
     - 98.0%
     - 98.5%
     - 92.9%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 97.1%
     - 97.2%
     - 98.0%
     - 94.8%
   * - Qwen3.5-9B
     - Data Default WWB
     - Similarity
     - 97.3%
     - 96.3%
     - 98.2%
     - 89.3%
   * - Qwen3.6-27B
     - Data Default WWB
     - Similarity
     - 98.2%
     - 94.4%
     - %
     - 94.2%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 97.1%
     - 94.9%
     - 93.6%
     - 99.4%
   * - LTX-VIDEO
     - Data Default WWB
     - Similarity
     - 90.7%
     - %
     - 75.9%
     - 82.1%

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
