.. {#openvino_docs_OV_UG_Samples_Overview}

OpenVINO™ Samples
===================


.. _code samples:

.. meta::
   :description: OpenVINO™ samples include a collection of simple console applications
                 that explain how to implement the capabilities and features of
                 OpenVINO API into an application.


.. toctree::
   :maxdepth: 1
   :hidden:

   Get Started with C++ Samples <openvino_docs_get_started_get_started_demos>
   openvino_sample_hello_classification
   openvino_sample_hello_nv12_input_classification
   openvino_sample_hello_query_device
   openvino_sample_hello_reshape_ssd
   openvino_sample_image_classification_async
   openvino_sample_model_creation
   openvino_sample_sync_benchmark
   openvino_sample_throughput_benchmark
   openvino_sample_bert_benchmark
   openvino_sample_benchmark_tool


The OpenVINO™ samples are simple console applications that show how to utilize
specific OpenVINO API capabilities within an application. They can assist you in
executing specific tasks such as loading a model, running inference, querying
specific device capabilities, etc.

The applications include:

.. important::

   All C++ samples support input paths containing only ASCII characters, except
   for the Hello Classification Sample, which supports Unicode.

- :doc:`Hello Classification Sample <openvino_sample_hello_classification>` -
  Inference of image classification networks like AlexNet and GoogLeNet using
  Synchronous Inference Request API. Input of any size and layout can be set to
  an infer request which will be pre-processed automatically during inference.
  The sample supports only images as input and supports input paths containing
  only Unicode characters.

- :doc:`Hello NV12 Input Classification Sample <openvino_sample_hello_nv12_input_classification>` -
  Input of any size and layout can be provided to an infer request. The sample
  transforms the input to the NV12 color format and pre-process it automatically
  during inference. The sample supports only images as input.

- :doc:`Hello Query Device Sample <openvino_sample_hello_query_device>` -
  Query of available OpenVINO devices and their metrics, configuration values.

- :doc:`Hello Reshape SSD Sample <openvino_sample_hello_reshape_ssd>` -
  Inference of SSD networks resized by ShapeInfer API according to an input size.

- :doc:`Image Classification Async Sample <openvino_sample_image_classification_async>` -
  Inference of image classification networks like AlexNet and GoogLeNet using
  Asynchronous Inference Request API. The sample supports only images as inputs.

- :doc:`OpenVINO Model Creation Sample <openvino_sample_model_creation>` -
  Construction of the LeNet model using the OpenVINO model creation sample.

- **Benchmark Samples** - Simple estimation of a model inference performance

  - :doc:`Sync Samples <openvino_sample_sync_benchmark>`
  - :doc:`Throughput Samples <openvino_sample_throughput_benchmark>`
  - :doc:`Bert Python Sample <openvino_sample_bert_benchmark>`

- :doc:`Benchmark Application <openvino_sample_benchmark_tool>` - Estimates deep
  learning inference performance on supported devices for synchronous and
  asynchronous modes.

  Python version of the benchmark tool is a core component of the OpenVINO
  installation package and may be executed with the following command:

  .. code-block:: console

     benchmark_app -m <model> -i <input> -d <device>


Additional Resources
####################

* :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
* :doc:`OpenVINO Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`
