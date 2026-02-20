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
     - -0.93%
     - -1.01%
     - -1.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.73%
     - 0.73%
     - 0.73%
     - 0.73%
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
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.00%
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
     - -2.18%
     - -2.18%
     - -2.18%
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
     - 98.1%
     - 94.1%
     - 99.6%
     - 94.0%
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - Data Default WWB
     - Similarity
     - 96.5%
     - 92.4%
     - 99.7%
     - 92.7%
   * - Gemma-3-4B-it
     - Data Default WWB
     - Similarity
     - 92.2%
     - 83.9%
     - 
     - 92.9%
   * - GPT-OSS-20B
     - Data Default WWB
     - Similarity
     - 94.9%
     - 92.2%
     - 
     - 92.9%
   * - Llama-2-7B-chat
     - Data Default WWB
     - Similarity
     - 99.3%
     - 93.3%
     - 99.6%
     - 93.3%
   * - Llama-3-8B
     - Data Default WWB
     - Similarity
     - 98.8%
     - 94.7%
     - 99.9%
     - 94.4%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 98.3%
     - 94.8%
     - 99.9%
     - 94.3%
   * - MiniCPM-V-2.6
     - Data Default WWB
     - Similarity
     - 90.6%
     - 90.1%
     - 88.1%
     - 89.1%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 95.1%
     - 92.5%
     - 99.1%
     - 92.1%
   * - Qwen2.5-VL-7B
     - Data Default WWB
     - Similarity
     - 93.7%
     - 90.7%
     - 99.8%
     - 89.9%
   * - Qwen3-8B
     - Data Default WWB
     - Similarity
     - 97.9%
     - 93.6%
     - 99.8%
     - 92.8%
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 95.4%
     - 96.1%
     - 
     - 95.1%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 96.7%
     - 95.5%
     - 99.5%
     - 92.1%

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
