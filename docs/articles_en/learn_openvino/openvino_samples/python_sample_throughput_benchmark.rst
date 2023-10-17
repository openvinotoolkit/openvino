.. {#openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README}

Throughput Benchmark Python Sample
==================================


.. meta::
   :description: Learn how to estimate performance of a model using Asynchronous Inference Request (Python) API in throughput mode.


This sample demonstrates how to estimate performance of a model using Asynchronous Inference Request API in throughput mode. Unlike :doc:`demos <omz_demos>` this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

The reported results may deviate from what :doc:`benchmark_app <openvino_inference_engine_tools_benchmark_tool_README>` reports. One example is model input precision for computer vision tasks. benchmark_app sets uint8, while the sample uses default model precision which is usually float32.

.. tab-set::

   .. tab-item:: Requirements 

      +--------------------------------+------------------------------------------------------------------------------+
      | Options                        | Values                                                                       |
      +================================+==============================================================================+
      | Validated Models               | :doc:`alexnet <omz_models_model_alexnet>`,                                   |
      |                                | :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`,                         |
      |                                | :doc:`yolo-v3-tf <omz_models_model_yolo_v3_tf>`,                             |
      |                                | :doc:`face-detection-0200 <omz_models_model_face_detection_0200>`            |
      +--------------------------------+------------------------------------------------------------------------------+
      | Model Format                   | OpenVINO™ toolkit Intermediate Representation                                |
      |                                | (\*.xml + \*.bin), ONNX (\*.onnx)                                            |
      +--------------------------------+------------------------------------------------------------------------------+
      | Supported devices              | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`         |
      +--------------------------------+------------------------------------------------------------------------------+
      | Other language realization     | :doc:`C++ <openvino_inference_engine_samples_throughput_benchmark_README>`   |
      +--------------------------------+------------------------------------------------------------------------------+

   .. tab-item:: Python API  

      The following Python API is used in the application:

      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Feature                        | API                                             | Description                                  |
      +================================+=================================================+==============================================+
      | OpenVINO Runtime Version       | [openvino.runtime.get_version]                  | Get Openvino API version.                    |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Basic Infer Flow               | [openvino.runtime.Core],                        | Common API to do inference: compile a model, |
      |                                | [openvino.runtime.Core.compile_model]           | configure input tensors.                     |
      |                                | [openvino.runtime.InferRequest.get_tensor]      |                                              |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Asynchronous Infer             | [openvino.runtime.AsyncInferQueue],             | Do asynchronous inference.                   |
      |                                | [openvino.runtime.AsyncInferQueue.start_async], |                                              |
      |                                | [openvino.runtime.AsyncInferQueue.wait_all],    |                                              |
      |                                | [openvino.runtime.InferRequest.results]         |                                              |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Model Operations               | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Tensor Operations              | [openvino.runtime.Tensor.get_shape],            | Get a tensor shape and its data.             |
      |                                | [openvino.runtime.Tensor.data]                  |                                              |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+

   .. tab-item:: Sample Code  

      .. doxygensnippet:: samples/python/benchmark/throughput_benchmark/throughput_benchmark.py
         :language: python

How It Works
####################

The sample compiles a model for a given device, randomly generates input data, performs asynchronous inference multiple times for a given number of seconds. Then processes and reports performance results.

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
####################

.. code-block:: sh

   python throughput_benchmark.py <path_to_model>


To run the sample, you need to specify a model:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.

.. note::

   Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.

   The sample accepts models in ONNX format (.onnx) that do not require preprocessing.


Example
++++++++++++++++++++

1. Install the ``openvino-dev`` Python package to use Open Model Zoo Tools:

   .. code-block:: sh

      python -m pip install openvino-dev[caffe]


2. Download a pre-trained model using:

   .. code-block:: sh

      omz_downloader --name googlenet-v1


3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

   .. code-block:: sh

      omz_converter --name googlenet-v1


4. Perform benchmarking using the ``googlenet-v1`` model on a ``CPU``:

   .. code-block:: sh

      python throughput_benchmark.py googlenet-v1.xml


Sample Output
####################

The application outputs performance results.

.. code-block:: sh

   [ INFO ] OpenVINO:
   [ INFO ] Build ................................. <version>
   [ INFO ] Count:          2817 iterations
   [ INFO ] Duration:       10012.65 ms
   [ INFO ] Latency:
   [ INFO ]     Median:     13.80 ms
   [ INFO ]     Average:    14.10 ms
   [ INFO ]     Min:        8.35 ms
   [ INFO ]     Max:        28.38 ms
   [ INFO ] Throughput: 281.34 FPS


See Also
####################

* :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
* :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Model Downloader <omz_tools_downloader>`
* :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

