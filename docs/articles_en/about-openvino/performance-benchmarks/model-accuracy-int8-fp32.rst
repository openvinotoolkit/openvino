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
     - 2.89%
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
     - -1.03%
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
     - -0.03%
     - 0.07%
   * - yolo_v11
     - COCO2017_detection_80cl
     - map
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
     - 0.02%
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
     - 0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.01%
     - -0.06%
   * - yolo_v11
     - COCO2017_detection_80cl
     - map
     - -0.03%
     - -2.21%
     - -2.21%
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
     - 23.8%
     - 27.3%
     - 
     - 23.06%
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - Data Default WWB
     - Similarity
     - 33.42%
     - 38.7%
     - 33.3%
     - 39.8%
   * - Gemma-2-9B-it
     - Data Default WWB
     - Similarity
     - 1.41%
     - 3.5%
     - %
     - 3.38%
   * - GLM4-9B-Chat
     - Data Default WWB
     - Similarity
     - 1.2%
     - 6.68%
     - 5.47%
     - 
   * - Qwen-2.5-1.5B-instruct
     - Data Default WWB
     - Similarity
     - 5.07%
     - 11.24%
     - 0.3
     - 12.77%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 2.35%
     - 5.99%
     - 1.3%
     - 5.84%
   * - Mistral-7b-instruct-V0.3
     - Data Default WWB
     - Similarity
     - 1.71%
     - 7.24%
     - 0.07%
     - 6.49%
   * - Phi4-mini-instruct
     - Data Default WWB
     - Similarity
     - 3.63%
     - 7.46%
     - 0.69%
     - 8.15%
   * - Qwen2-VL-7B
     - Data Default WWB
     - Similarity
     - 6.12%
     - 7.89%
     - 4.09%
     - 8.52%
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 4.67%
     - 3.85%
     - 
     - 3.45%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 3.29%
     - 4.91%
     - 0.50%
     - 9.16%

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
