# Performance Information F.A.Q. {#openvino_docs_performance_benchmarks_faq}


@sphinxdirective


.. meta::
   :description: Check the F.A.Q. for performance benchmarks in Intel® Distribution of OpenVINO™ toolkit.




.. dropdown:: How often do performance benchmarks get updated?

   New performance benchmarks are typically published on every
   ``major.minor`` release of the Intel® Distribution of OpenVINO™ toolkit.

.. dropdown:: Where can I find the models used in the performance benchmarks?

   All models used are included in the GitHub repository of :doc:`Open Model Zoo <model_zoo>`.

.. dropdown:: Will there be any new models added to the list used for benchmarking?

   The models used in the performance benchmarks were chosen based
   on general adoption and usage in deployment scenarios. New models that
   support a diverse set of workloads and usage are added periodically.

.. dropdown:: How can I run the benchmark results on my own?

   All of the performance benchmarks are generated using the
   open-source tool within the Intel® Distribution of OpenVINO™ toolkit
   called ``benchmark_app``. This tool is available 
   :doc:`for C++ apps <openvino_inference_engine_samples_benchmark_app_README>`.
   as well as 
   :doc:`for Python apps <openvino_inference_engine_tools_benchmark_tool_README>`.

   For a simple instruction on testing performance, see the :doc:`Getting Performance Numbers Guide <openvino_docs_MO_DG_Getting_Performance_Numbers>`.

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
      * - `chatGLM2-6B <https://huggingface.co/THUDM/chatglm2-6b/tree/main>`__
        - THUDM
        - Transformer
        - 32K
      * - `Llama-2-7b-chat <https://ai.meta.com/llama/>`__
        - Meta AI
        - Auto regressive language
        - 4096
      * - `Stable-Diffusion-V2-1 <https://huggingface.co/stabilityai/stable-diffusion-2-1>`__
        - Hugginface
        - Latent Diffusion Model
        - 77
      * - `bert-base-cased <https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1>`__
        - BERT
        - question / answer
        - 128
      * - `bert-large-uncased-whole-word-masking-squad-int8-0001 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-large-uncased-whole-word-masking-squad-int8-0001>`__
        - BERT-large
        - question / answer
        - 384
      * - `deeplabv3 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3>`__
        -  DeepLab v3 Tf
        - semantic segmentation
        - 513x513
      * - `efficientdet-d0 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientdet-d0-tf>`__
        - Efficientdet
        - classification
        - 512x512
      * - `faster_rcnn_resnet50_coco <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_resnet50_coco>`__
        - Faster RCNN TF
        - object detection
        - 600x1024
      * - `mobilenet-v2 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-pytorch>`__
        - Mobilenet V2 PyTorch
        - classification
        - 224x224
      * - `resnet-50 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf>`__
        - ResNet-50_v1_ILSVRC-2012
        - classification
        - 224x224
      * - `ssd-mobilenet-v1-coco <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco>`__
        - ssd-mobilenet-V1-coco onnx model
        - object detection
        - 300x300      
      * - `ssd-resnet34-1200-onnx <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd-resnet34-1200-onnx>`__
        - ssd-resnet34 onnx model
        - object detection
        - 1200x1200      
      * - `unet-camvid-onnx-0001 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/unet-camvid-onnx-0001>`__
        - U-Net
        - semantic segmentation
        - 368x480     
      * - `yolo-v3 <https://https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf>`__
        - YOLO v3 
        - object detection
        - 416x416      
      * - `yolo-v3-tiny <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tiny-tf>`__
        - YOLO v3 Tiny
        - object detection
        - 416x416      
      * - `yolov8n <https://https://github.com/ultralytics/ultralytics>`__
        - Yolov8nano 
        -  object detection
        - 608x608


.. dropdown:: Where can I purchase the specific hardware used in the benchmarking?

   Intel partners with vendors all over the world. For a list of Hardware Manufacturers, see the 
   `Intel® AI: In Production Partners & Solutions Catalog <https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html>`__. 
   For more details, see the :doc:`Supported Devices <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`.
   documentation. Before purchasing any hardware, you can test and run
   models remotely, using `Intel® DevCloud for the Edge <http://devcloud.intel.com/edge/>`__.

.. dropdown:: How can I optimize my models for better performance or accuracy?

   Set of guidelines and recommendations to optimize models are available in the 
   :doc:`optimization guide <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`.
   Join the conversation in the `Community Forum <https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit>`__ for further support.

.. dropdown:: Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?

   The benefit of low-precision optimization using the OpenVINO™
   toolkit model optimizer extends beyond processors supporting VNNI
   through Intel® DL Boost. The reduced bit width of INT8 compared to FP32
   allows Intel® CPU to process the data faster. Therefore, it offers
   better throughput on any converted model, regardless of the
   intrinsically supported low-precision optimizations within Intel®
   hardware. For comparison on boost factors for different network models
   and a selection of Intel® CPU architectures, including AVX-2 with Intel®
   Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel®
   Xeon® 8270, refer to the :doc:`Model Accuracy for INT8 and FP32 Precision <openvino_docs_performance_int8_vs_fp32>`

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


@endsphinxdirective