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
     - 2.41%
     - 2.78%
     - 2.61%
     - 2.84%
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 
     - 
     - 
     - 
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - -1.03%
     - -1.00%
     - -1.03%
     - -1.01%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - -0.17%
     - -0.17%
     - -0.18%
     - -0.17%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - -0.01%
     - -0.01%
     - -0.04%
     - -0.04%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - -0.09%
     - -0.09%
     - -0.02%
     - -0.04%
.. list-table:: Model Accuracy for BF16, FP32 and FP16 (FP16: Arc only. BF16: Xeon® 6972P only)
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
   * - mask_rcnn_resnet50_atrous_coco
     - COCO2017_detection_91cl_bkgr
     - coco_orig_precision
     - 
     - 
     - 
     - 
     - 
   * - mobilenet-v2
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - -0.23%
     - -0.03%
   * - resnet-50
     - ImageNet2012
     - accuracy @ top1
     - 0.00%
     - 0.00%
     - 0.00%
     - 0.06%
     - 0.01%
   * - ssd-resnet34-1200
     - COCO2017_detection_80cl_bkgr
     - map
     - 0.02%
     - 0.02%
     - 0.01%
     - 0.02%
     - 0.06%
   * - yolo_v8n
     - COCO2017_detection_80cl
     - map
     - 0.01%
     - 0.01%
     - 0.01%
     - 
     - -0.03%
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
     - 10.3%
     - 21.4%
     - 0.21%
     - 23.5%
   * - DeepSeek-R1-Distill-Qwen-1.5B
     - Data Default WWB
     - Similarity
     - 16.1%
     - 34.5%
     - 2.48%
     - 36.4%
   * - DeepSeek-R1-Distill-Qwen-7B
     - Data Default WWB
     - Similarity
     - 25.5%
     - 35.6%
     - 3.9%
     - 37.2%
   * - GLM4-9B-Chat
     - Data Default WWB
     - Similarity
     - 6.9%
     - 3.8%
     - 6.3%
     - 15.1%
   * - Qwen-2.5-7B-instruct
     - Data Default WWB
     - Similarity
     - 7.97%
     - 25.12%
     - 0.09%
     - 23.87%
   * - Gemma-2-9B
     - Data Default WWB
     - Similarity
     - 4.81%
     - 10.25%
     - 1.73%
     - 10.24%
   * - Llama-2-7b-chat
     - Data Default WWB
     - Similarity
     - 1.80%
     - 22.31%
     - 0.13%
     - 21.54%
   * - Llama-3-8b
     - Data Default WWB
     - Similarity
     - 2.26%
     - 23.00%
     - 0.12%
     - 23.59%
   * - Llama-3.2-3b-instruct
     - Data Default WWB
     - Similarity
     - 2.40%
     - 11.25%
     - 0.00%
     - 12.32%
   * - Mistral-7b-instruct-V0.2
     - Data Default WWB
     - Similarity
     - 2.94%
     - 9.08%
     - 0.37%
     - 9.53%
   * - Phi3-mini-4k-instruct
     - Data Default WWB
     - Similarity
     - 8.08%
     - 7.93%
     - 0.00%
     - 8.30%
   * - Qwen-2-7B
     - Data Default WWB
     - Similarity
     - 4.97%
     - 18.97%
     - 0.00%
     - 22.38%
   * - Flux.1-schnell
     - Data Default WWB
     - Similarity
     - 4.60%
     - 4.20%
     - 5.00%
     - 3.30%
   * - Stable-Diffusion-V1-5
     - Data Default WWB
     - Similarity
     - 2.50%
     - 1.90%
     - 2.10%
     - 0.10%

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
