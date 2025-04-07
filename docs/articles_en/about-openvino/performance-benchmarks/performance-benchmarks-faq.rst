Performance Information F.A.Q.
==============================


.. meta::
   :description: Check the F.A.Q. for performance benchmarks in Intel® Distribution of OpenVINO™ toolkit.




.. dropdown:: How often do performance benchmarks get updated?

   New performance benchmarks are typically published on every
   ``major.minor`` release of the Intel® Distribution of OpenVINO™ toolkit.

.. dropdown:: Where can I find the models used in the performance benchmarks?

   All models used are published on `Hugging Face <https://huggingface.co/OpenVINO>`__.

.. dropdown:: Will there be any new models added to the list used for benchmarking?

   The models used in the performance benchmarks were chosen based
   on general adoption and usage in deployment scenarios. New models that
   support a diverse set of workloads and usage are added periodically.

.. dropdown:: How can I run the benchmark results on my own?

   All of the performance benchmarks on conventional network models are generated using the
   open-source tool within the Intel® Distribution of OpenVINO™ toolkit
   called :doc:`benchmark_app <../../get-started/learn-openvino/openvino-samples/benchmark-tool>`.

   For diffusers (Stable-Diffusion) and foundational models (aka LLMs) please use the OpenVINO GenAI
   opensource repo `OpenVINO GenAI tools/llm_bench <https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench>`__

   For a simple instruction on testing performance, see the :doc:`Getting Performance Numbers Guide <getting-performance-numbers>`.

.. dropdown:: Where can I find a more detailed description of the workloads used for benchmarking?

   The image size used in inference depends on the benchmarked
   network. The table below presents the list of input sizes for each
   network model and a link to more information on that model:

   .. list-table::
      :header-rows: 1

      * - Model
        - Public Network
        - Task
        - Input Size
      * - `DeepSeek-R1-Distill-Llama-8B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`__
        - DeepSeek, HF
        - Auto regressive language
        - 128K
      * - `DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__
        - DeepSeek, HF
        - Auto regressive language
        - 128K
      * - `DeepSeek-R1-Distill-Qwen-7B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__
        - DeepSeek, HF
        - Auto regressive language
        - 128K
      * - `GLM4-9B-chat <https://huggingface.co/THUDM/glm-4-9b-chat/tree/main>`__
        - THUDM
        - Transformer
        - 128K
      * - `Gemma-2-9B <https://huggingface.co/google/gemma-2-9b-it>`__
        - Hugginface
        - Text-To-Text Decoder-only
        - 8K
      * - `Llama-2-7b-chat <https://www.llama.com/>`__
        - Meta AI
        - Auto regressive language
        - 4K
      * - `Llama-3-8b <https://www.llama.com/>`__
        - Meta AI
        - Auto regressive language
        - 8K
      * - `Llama-3.2-3B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`__
        - Meta AI
        - Auto regressive language
        - 128K
      * - `Mistral-7b-Instruct-V0.2 <https://huggingface.co/mistralai/Mistral-7B-v0.2>`__
        - Mistral AI
        - Auto regressive language
        - 32K
      * - `Phi3-4k-mini-Instruct <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__
        - Huggingface
        - Auto regressive language
        - 4096
      * - `Qwen-2-7B <https://huggingface.co/Qwen/Qwen2-7B>`__
        - Huggingface
        - Auto regressive language
        - 128K
      * - `Qwen-2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__
        - Huggingface
        - Auto regressive language
        - 128K
      * - `Stable-Diffusion-V1-5 <https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5>`__
        - Hugginface
        - Latent Diffusion Model
        - 77
      * - `FLUX.1-schnell <https://huggingface.co/black-forest-labs/FLUX.1-schnell>`__
        - Hugginface
        - Latent Adversarial Diffusion Distillation Model
        - 256
      * - `bert-base-cased <https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1>`__
        - BERT
        - question / answer
        - 128
      * - `mask_rcnn_resnet50_atrous_coco <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mask_rcnn_resnet50_atrous_coco>`__
        - Mask R-CNN ResNet 50 Atrous
        - object instance segmentation
        - 800x1365
      * - `mobilenet-v2 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-pytorch>`__
        - Mobilenet V2 PyTorch
        - classification
        - 224x224
      * - `resnet-50 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf>`__
        - ResNet-50_v1_ILSVRC-2012
        - classification
        - 224x224
      * - `ssd-resnet34-1200-onnx <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd-resnet34-1200-onnx>`__
        - ssd-resnet34 onnx model
        - object detection
        - 1200x1200
      * - `yolov8n <https://github.com/ultralytics/ultralytics>`__
        - Yolov8nano
        - object detection
        - 608x608

.. dropdown:: Where can I purchase the specific hardware used in the benchmarking?

   Intel partners with vendors all over the world. For a list of Hardware Manufacturers, see the
   `Intel® AI: In Production Partners & Solutions Catalog <https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html>`__.
   For more details, see the :doc:`Supported Devices <../../documentation/compatibility-and-support/supported-devices>` article.


.. dropdown:: How can I optimize my models for better performance or accuracy?

   Set of guidelines and recommendations to optimize models are available in the
   :doc:`optimization guide <../../openvino-workflow/running-inference/optimize-inference>`.
   Join the conversation in the `Community Forum <https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit>`__ for further support.

.. dropdown:: Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?

   The benefit of low-precision optimization extends beyond processors supporting VNNI
   through Intel® DL Boost. The reduced bit width of INT8 compared to FP32
   allows Intel® CPU to process the data faster. Therefore, it offers
   better throughput on any converted model, regardless of the
   intrinsically supported low-precision optimizations within Intel®
   hardware. For comparison on boost factors for different network models
   and a selection of Intel® CPU architectures, including AVX-2 with Intel®
   Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel®
   Xeon® 8270, refer to the :doc:`Model Accuracy for INT8 and FP32 Precision <model-accuracy-int8-fp32>`

.. dropdown:: Where can I search for OpenVINO™ performance results based on HW-platforms?

   The website format has changed in order to support more common
   approach of searching for the performance results of a given neural
   network model on different HW-platforms. As opposed to reviewing
   performance of a given HW-platform when working with different neural
   network models.

.. dropdown:: How is Latency measured?

   Latency is measured by running the OpenVINO™ Runtime in
   synchronous mode. In this mode, each frame or image is processed through
   the entire set of stages (pre-processing, inference, post-processing)
   before the next frame or image is processed. This KPI is relevant for
   applications where the inference on a single image is required. For
   example, the analysis of an ultra sound image in a medical application
   or the analysis of a seismic image in the oil & gas industry. Other use
   cases include real or near real-time applications, e.g. the response of
   industrial robot to changes in its environment and obstacle avoidance
   for autonomous vehicles, where a quick response to the result of the
   inference is required.



.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see:
   :doc:`Platforms, Configurations, Methodology <../performance-benchmarks>`,
   :doc:`Legal Information <../additional-resources/terms-of-use>`.