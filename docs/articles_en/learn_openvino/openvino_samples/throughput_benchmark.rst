.. {#openvino_sample_throughput_benchmark}

Throughput Benchmark Sample
===========================


.. meta::
   :description: Learn how to estimate performance of a model using Asynchronous Inference Request API in throughput mode (Python, C++).


This sample demonstrates how to estimate performance of a model using Asynchronous 
Inference Request API in throughput mode. Unlike :doc:`demos <omz_demos>` this sample 
does not have other configurable command-line arguments. Feel free to modify sample's 
source code to try out different options.

The reported results may deviate from what :doc:`benchmark_app <openvino_sample_benchmark_tool>` 
reports. One example is model input precision for computer vision tasks. benchmark_app 
sets ``uint8``, while the sample uses default model precision which is usually ``float32``.

Requirements
####################

+-------------------+----------------------------------------------------------------------+
| Options           | Values                                                               |
+===================+======================================================================+
| Validated Models  | :doc:`alexnet <omz_models_model_alexnet>`,                           |
|                   | :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`,                 |
|                   | :doc:`yolo-v3-tf <omz_models_model_yolo_v3_tf>`,                     |
|                   | :doc:`face-detection-0200 <omz_models_model_face_detection_0200>`    |
+-------------------+----------------------------------------------------------------------+
| Model Format      | OpenVINO™ toolkit Intermediate Representation                        |
|                   | (\*.xml + \*.bin), ONNX (\*.onnx)                                    |
+-------------------+----------------------------------------------------------------------+
| Supported devices | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>` |
+-------------------+----------------------------------------------------------------------+


How It Works
####################

The sample compiles a model for a given device, randomly generates input data, 
performs asynchronous inference multiple times for a given number of seconds. 
Then, it processes and reports performance results.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/python/benchmark/throughput_benchmark/throughput_benchmark.py
                  :language: python

         .. tab-item:: API

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


   .. tab-item:: C++
      :sync: cpp

      .. tab-set::
      
         .. tab-item:: Sample Code
      
            .. scrollbox::

               .. doxygensnippet:: samples/cpp/benchmark/throughput_benchmark/main.cpp
                  :language: cpp

         .. tab-item:: API
            
            The following C++ API is used in the application:
      
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Feature                  | API                                          | Description                                  |
            +==========================+==============================================+==============================================+
            | OpenVINO Runtime Version | ``ov::get_openvino_version``                 | Get Openvino API version.                    |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Basic Infer Flow         | ``ov::Core``, ``ov::Core::compile_model``,   | Common API to do inference: compile a model, |
            |                          | ``ov::CompiledModel::create_infer_request``, | create an infer request,                     |
            |                          | ``ov::InferRequest::get_tensor``             | configure input tensors.                     |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Asynchronous Infer       | ``ov::InferRequest::start_async``,           | Do asynchronous inference with callback.     |
            |                          | ``ov::InferRequest::set_callback``           |                                              |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Model Operations         | ``ov::CompiledModel::inputs``                | Get inputs of a model.                       |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Tensor Operations        | ``ov::Tensor::get_shape``,                   | Get a tensor shape and its data.             |
            |                          | ``ov::Tensor::data``                         |                                              |
            +--------------------------+----------------------------------------------+----------------------------------------------+
      

You can see the explicit description of each sample step at 
:doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` 
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
####################

To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>` section in OpenVINO™ Toolkit Samples guide.

Running
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console
      
         python throughput_benchmark.py <path_to_model> <device_name>(default: CPU)


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console
      
         throughput_benchmark <path_to_model> <device_name>(default: CPU)


To run the sample, you need to specify a model. You can get a model specific for 
your inference task from one of model repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.

.. note::

   Before running the sample with a trained model, make sure the model is converted 
   to the intermediate representation (IR) format (\*.xml + \*.bin) using 
   :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.

   The sample accepts models in ONNX format (.onnx) that do not require preprocessing.


Example
++++++++++++++++++++

1. Download a pre-trained model.
2. If a model is not in the IR or ONNX format, it must be converted by using:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            import openvino as ov

            ov_model = ov.convert_model('./models/googlenet-v1')
            # or, when model is a Python model object
            ov_model = ov.convert_model(googlenet-v1)

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: console

            ovc ./models/googlenet-v1

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            mo --input_model ./models/googlenet-v1


3. Perform benchmarking, using the ``googlenet-v1`` model on a ``CPU``:

   .. tab-set::
   
      .. tab-item:: Python
         :sync: python
   
         .. code-block:: console
      
            python throughput_benchmark.py ./models/googlenet-v1.xml
   
      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console
      
            throughput_benchmark ./models/googlenet-v1.xml


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The application outputs performance results.
      
      .. code-block:: console
      
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

   .. tab-item:: C++
      :sync: cpp

      The application outputs performance results.
      
      .. code-block:: console
      
         [ INFO ] OpenVINO:
         [ INFO ] Build ................................. <version>
         [ INFO ] Count:      1577 iterations
         [ INFO ] Duration:   15024.2 ms
         [ INFO ] Latency:
         [ INFO ]        Median:     38.02 ms
         [ INFO ]        Average:    38.08 ms
         [ INFO ]        Min:        25.23 ms
         [ INFO ]        Max:        49.16 ms
         [ INFO ] Throughput: 104.96 FPS


Additional Resources
####################

* :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
* :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
