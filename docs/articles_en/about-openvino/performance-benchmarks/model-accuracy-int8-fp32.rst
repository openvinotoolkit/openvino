Model Accuracy
==============



The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures. The third table presents the GenAI model accuracies as absolute accuracy values. Please also refer to notes below
the table for more information.

* A - Intel® Core™ Ultra 9-185H (AVX2), INT8 and FP32
* B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
* C - Intel® Xeon 6972P (VNNI, AMX), INT8, BF16, FP32
* D - Intel® Arc-B580, INT8 and FP16


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
     - 2.63%
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
     - -0.93%
     - -0.93%
     - -0.91%
     - -1.03%
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
     - 0.00%
     - 0.03%
     - 0.07%
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
     - 0.00%
     - 0.00%
     - 0.02%
     - 0.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.04%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.02%
     - 0.06%
   * - yolo_v11
     - COCO2017_detection_80cl
     - AP@0.5:0.05:0.95
     - 0.00%
     - 0.00%
     - 0.00%
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
     - 98.1%
     - 94.4%
     - 99.5%
     - 92.6%
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - Data Default WWB
     - Similarity
     - 96.5%
     - 92.4%
     - 99.7%
     - 92.1%
   * - Gemma-3-1B-it
     - Data Default WWB
     - Similarity
     - 97.3%
     - 92.0%
     - 99.2%
     - 91.5%
   * - GLM4-9B-Chat
     - Data Default WWB
     - Similarity
     - 98.8%
     - 93.3%
     - %
     - 95.0%
   * - Llama-2-7B-chat
     - Data Default WWB
     - Similarity
     - 99.3%
     - 93.4%
     - 99.8%
     - 91.9%
   * - Llama-3-8B
     - Data Default WWB
     - Similarity
     - 98.8%
     - 94.3%
     - %
     - 94.5%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 98.2%
     - 93.2%
     - 98.4%
     - 94.0%
   * - Mistral-7b-instruct-V0.3
     - Data Default WWB
     - Similarity
     - 98.3%
     - 92.8%
     - 99.9%
     - 93.6%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 96.4%
     - 92.0%
     - 99.3%
     - 91.7%
   * - Qwen2-VL-7B
     - Data Default WWB
     - Similarity
     - 97.8%
     - 92.4%
     - 99.8%
     - 93.0%
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 95.4%
     - 96.1%
     - 
     - 92.1%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 97.3%
     - 95.1%
     - 99.5%
     - 91.5%

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
