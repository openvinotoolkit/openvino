Performance Benchmarks
======================

.. meta::
   :description: Use the benchmark results for Intel® Distribution of OpenVINO™
                 toolkit, that may help you decide what hardware to use or how
                 to plan the workload.

.. toctree::
   :maxdepth: 1
   :hidden:

   Efficient LLMs for AI PC <performance-benchmarks/generative-ai-performance>
   Performance Information F.A.Q. <performance-benchmarks/performance-benchmarks-faq>
   OpenVINO Accuracy <performance-benchmarks/model-accuracy-int8-fp32>
   Getting Performance Numbers <performance-benchmarks/getting-performance-numbers>


This page presents benchmark results for the
`Intel® Distribution of OpenVINO™ toolkit <https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html>`__
and :doc:`OpenVINO Model Server <../openvino-workflow/model-server/ovms_what_is_openvino_model_server>`, for a representative
selection of public neural networks and Intel® devices. The results may help you decide which
hardware to use in your applications or plan AI workload for the hardware you have already
implemented in your solutions. Click the buttons below to see the chosen benchmark data.

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. button-link:: #
         :class: ov-toolkit-benchmark-results
         :color: primary
         :outline:
         :expand:

         :material-regular:`bar_chart;1.4em` OpenVINO Benchmark Graphs (general)

   .. grid-item::

      .. button-link:: #
         :class: ovms-toolkit-benchmark-results
         :color: primary
         :outline:
         :expand:

         :material-regular:`bar_chart;1.4em` OVMS Benchmark Graphs (general)

   .. grid-item::

      .. button-link:: ./performance-benchmarks/generative-ai-performance.html
         :class: ov-toolkit-benchmark-genai
         :color: primary
         :outline:
         :expand:

         :material-regular:`table_view;1.4em` LLM performance for AI PC

   .. grid-item::

      .. button-link:: #
         :class: ovms-toolkit-benchmark-llm
         :color: primary
         :outline:
         :expand:

         :material-regular:`bar_chart;1.4em` OVMS for GenAI (coming soon)







**Key performance indicators and workload parameters**

.. tab-set::

   .. tab-item:: Throughput
      :sync: throughput

      For Vision and NLP Models this measures the number of inferences delivered within a latency threshold
      (for example, number of Frames Per Second - FPS).
      For GenAI (or Large Language Models) this measures the token rate after the first token aka. 2nd token
      throughput rate which is presented as tokens/sec. Please click on the "Workload Parameters" tab to
      learn more about input/output token lengths, etc.

   .. tab-item:: Latency
      :sync: latency

      For Vision and NLP models this measures the synchronous execution of inference requests and
      is reported in milliseconds. Each inference request (for example: preprocess, infer,
      postprocess) is allowed to complete before the next one starts. This performance metric is
      relevant in usage scenarios where a single image input needs to be acted upon as soon as
      possible. An example would be the healthcare sector where medical personnel only request
      analysis of a single ultra sound scanning image or in real-time or near real-time applications
      such as an industrial robot's response to actions in its environment or obstacle avoidance
      for autonomous vehicles.
      For Transformer models like Stable-Diffusion this measures the time it takes to convert the prompt
      or input text into a finished image. It is presented in seconds.

   .. tab-item:: Workload Parameters
      :sync: workloadparameters

      The workload parameters affect the performance results of the different models we use for
      benchmarking. Image processing models have different image size definitions and the
      Natural Language Processing models have different max token list lengths. All these can
      be found in detail in the :doc:`FAQ section <performance-benchmarks/performance-benchmarks-faq>`.
      All models are executed using a batch size of 1. Below are the parameters for the GenAI
      models we display.

      * Input tokens: 1024,
      * Output tokens: 128,
      * number of beams: 1

      For text to image:

      * iteration steps: 20,
      * image size (HxW): 256 x 256,
      * input token length: 1024 (the tokens for GenAI models are in English).


**Platforms, Configurations, Methodology**

To see the methodology used to obtain the numbers and learn how to test performance yourself,
see the guide on :doc:`getting performance numbers <performance-benchmarks/getting-performance-numbers>`.

For a listing of all platforms and configurations used for testing, refer to the following:

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. button-link:: ../_static/benchmarks_files/OV-2024.4-platform_list.pdf
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Click for Hardware Platforms [PDF]

      .. button-link:: ../_static/benchmarks_files/OV-2024.4-system-info-detailed.xlsx
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Click for Configuration Details [XLSX]

      .. button-link:: ../_static/benchmarks_files/OV-2024.4-Performance-Data.xlsx
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Click for Performance Data [XLSX]





**Disclaimers**

* Intel® Distribution of OpenVINO™ toolkit performance results are based on release
  2024.3, as of July 31, 2024.

* OpenVINO Model Server performance results are based on release
  2024.3, as of Aug. 19, 2024.

The results may not reflect all publicly available updates. Intel technologies' features and
benefits depend on system configuration and may require enabled hardware, software, or service
activation. Learn more at intel.com, the OEM, or retailer.

See configuration disclosure for details. No product can be absolutely secure.
Performance varies by use, configuration and other factors. Learn more at
`www.intel.com/PerformanceIndex <https://www.intel.com/PerformanceIndex>`__.
Intel optimizations, for Intel compilers or other products, may not optimize to the same degree
for non-Intel products.





.. raw:: html

   <link rel="stylesheet" type="text/css" href="../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see
   :doc:`F.A.Q. <./performance-benchmarks/performance-benchmarks-faq>`
   See :doc:`Legal Information <./additional-resources/terms-of-use>`.
