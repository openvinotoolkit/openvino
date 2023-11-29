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
   openvino_inference_engine_samples_classification_sample_async_README
   openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README
   openvino_inference_engine_samples_hello_classification_README
   openvino_inference_engine_ie_bridges_c_samples_hello_classification_README
   openvino_inference_engine_ie_bridges_python_sample_hello_classification_README
   openvino_inference_engine_samples_hello_reshape_ssd_README
   openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README
   openvino_inference_engine_samples_hello_nv12_input_classification_README
   openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README
   openvino_inference_engine_samples_hello_query_device_README
   openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README
   openvino_inference_engine_samples_model_creation_sample_README
   openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README
   openvino_inference_engine_samples_speech_sample_README
   openvino_inference_engine_ie_bridges_python_sample_speech_sample_README
   openvino_inference_engine_samples_sync_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_sync_benchmark_README
   openvino_inference_engine_samples_throughput_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README
   openvino_inference_engine_samples_benchmark_app_README
   openvino_inference_engine_tools_benchmark_tool_README


The OpenVINO™ samples are simple console applications that show how to utilize specific OpenVINO API capabilities within an application. They can assist you in executing specific tasks such as loading a model, running inference, querying specific device capabilities, etc.

The applications include:

.. important::
   
   All C++ samples support input paths containing only ASCII characters, except for the Hello Classification Sample, which supports Unicode.

- **Hello Classification Sample** – Inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference. The sample supports only images as input and supports input paths containing only Unicode characters.

  - :doc:`Python Sample <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>`
  - :doc:`C++ Sample <openvino_inference_engine_samples_hello_classification_README>`
  - :doc:`C Sample <openvino_inference_engine_ie_bridges_c_samples_hello_classification_README>`

- **Hello NV12 Input Classification Sample** – Input of any size and layout can be provided to an infer request. The sample transforms the input to the NV12 color format and pre-process it automatically during inference. The sample supports only images as input.

  - :doc:`C++ Sample <openvino_inference_engine_samples_hello_nv12_input_classification_README>`
  - :doc:`C Sample <openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README>`

- **Hello Query Device Sample** – Query of available OpenVINO devices and their metrics, configuration values.

  - :doc:`Python* Sample <openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README>`
  - :doc:`C++ Sample <openvino_inference_engine_samples_hello_query_device_README>`

- **Hello Reshape SSD Sample** – Inference of SSD networks resized by ShapeInfer API according to an input size.

  - :doc:`Python Sample** <openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README>`
  - :doc:`C++ Sample** <openvino_inference_engine_samples_hello_reshape_ssd_README>`

- **Image Classification Async Sample** – Inference of image classification networks like AlexNet and GoogLeNet using Asynchronous Inference Request API. The sample supports only images as inputs.

  - :doc:`Python* Sample <openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README>`
  - :doc:`C++ Sample <openvino_inference_engine_samples_classification_sample_async_README>`

- **OpenVINO Model Creation Sample** – Construction of the LeNet model using the OpenVINO model creation sample.

  - :doc:`Python Sample <openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README>`
  - :doc:`C++ Sample <openvino_inference_engine_samples_model_creation_sample_README>`

- **Benchmark Samples** - Simple estimation of a model inference performance

  - :doc:`Sync Python* Sample <openvino_inference_engine_ie_bridges_python_sample_sync_benchmark_README>`
  - :doc:`Sync C++ Sample <openvino_inference_engine_samples_sync_benchmark_README>`
  - :doc:`Throughput Python* Sample <openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README>`
  - :doc:`Throughput C++ Sample <openvino_inference_engine_samples_throughput_benchmark_README>`
  - :doc:`Bert Python* Sample <openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README>`

- **Benchmark Application** – Estimates deep learning inference performance on supported devices for synchronous and asynchronous modes.

  - :doc:`Benchmark Python Tool <openvino_inference_engine_tools_benchmark_tool_README>`

    - Python version of the benchmark tool is a core component of the OpenVINO installation package and 
      may be executed with the following command: ``benchmark_app -m <model> -i <input> -d <device>``. 
  - :doc:`Benchmark C++ Tool <openvino_inference_engine_samples_benchmark_app_README>`  


- **Automatic Speech Recognition Sample** - ``[DEPRECATED]`` Acoustic model inference based on Kaldi neural networks and speech feature vectors.

  - :doc:`Python Sample <openvino_inference_engine_ie_bridges_python_sample_speech_sample_README>`
  - :doc:`C++ Sample <openvino_inference_engine_samples_speech_sample_README>`


See Also
########

* :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
* :doc:`OpenVINO Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`


