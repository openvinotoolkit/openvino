.. {#openvino_sample_sync_benchmark}

Sync Benchmark Sample
=====================


.. meta::
   :description: Learn how to estimate performance of a model using Synchronous Inference Request API (Python, C++).


This sample demonstrates how to estimate performance of a model using Synchronous 
Inference Request API. It makes sense to use synchronous inference only in latency 
oriented scenarios. Models with static input shapes are supported. Unlike 
:doc:`demos <omz_demos>` this sample does not have other configurable command-line 
arguments. Feel free to modify sample's source code to try out different options.
Before using the sample, refer to the following requirements:

- The sample accepts models in OpenVINO Intermediate Representation (.xml + .bin) 
  and ONNX (.onnx) formats, that do not require preprocessing.
- The sample has been validated with: :doc:`alexnet <omz_models_model_alexnet>`, 
  :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`, :doc:`yolo-v3-tf <omz_models_model_yolo_v3_tf>`,
  :doc:`face-detection-0200 <omz_models_model_face_detection_0200>` models.
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>` 
  section in "Get Started with Samples" guide.

How It Works
####################

The sample compiles a model for a given device, randomly generates input data, 
performs synchronous inference multiple times for a given number of seconds. 
Then, it processes and reports performance results.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::
      
         .. tab-item:: Sample Code
      
            .. scrollbox::

               .. doxygensnippet:: samples/python/benchmark/sync_benchmark/sync_benchmark.py
                  :language: python

         .. tab-item:: API
      
            The following Python API is used in the application:
      
            +--------------------------------+-------------------------------------------------+----------------------------------------------+
            | Feature                        | API                                             | Description                                  |
            +================================+=================================================+==============================================+
            | OpenVINO Runtime Version       | [openvino.runtime.get_version]                  | Get Openvino API version.                    |
            +--------------------------------+-------------------------------------------------+----------------------------------------------+
            | Basic Infer Flow               | [openvino.runtime.Core],                        | Common API to do inference: compile a model, |
            |                                | [openvino.runtime.Core.compile_model],          | configure input tensors.                     |
            |                                | [openvino.runtime.InferRequest.get_tensor]      |                                              |
            +--------------------------------+-------------------------------------------------+----------------------------------------------+
            | Synchronous Infer              | [openvino.runtime.InferRequest.infer],          | Do synchronous inference.                    |
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

               .. doxygensnippet:: samples/cpp/benchmark/sync_benchmark/main.cpp
                  :language: cpp

         .. tab-item:: API
      
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Feature                  | API                                          | Description                                  |
            +==========================+==============================================+==============================================+
            | OpenVINO Runtime Version | ``ov::get_openvino_version``                 | Get Openvino API version.                    |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Basic Infer Flow         | ``ov::Core``, ``ov::Core::compile_model``,   | Common API to do inference: compile a model, |
            |                          | ``ov::CompiledModel::create_infer_request``, | create an infer request,                     |
            |                          | ``ov::InferRequest::get_tensor``             | configure input tensors.                     |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Synchronous Infer        | ``ov::InferRequest::infer``,                 | Do synchronous inference.                    |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Model Operations         | ``ov::CompiledModel::inputs``                | Get inputs of a model.                       |
            +--------------------------+----------------------------------------------+----------------------------------------------+
            | Tensor Operations        | ``ov::Tensor::get_shape``,                   | Get a tensor shape and its data.             |
            |                          | ``ov::Tensor::data``                         |                                              |
            +--------------------------+----------------------------------------------+----------------------------------------------+
      

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` 
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
####################


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console
      
         python sync_benchmark.py <path_to_model> <device_name>(default: CPU)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console
      
         sync_benchmark <path_to_model> <device_name>(default: CPU)


To run the sample, you need to specify a model. You can get a model specific for 
your inference task from one of model repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.

.. note::

   Before running the sample with a trained model, make sure the model is converted 
   to the OpenVINO Intermediate Representation (IR) format (\*.xml + \*.bin) using the 
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
      
            python sync_benchmark.py googlenet-v1.xml
   
      .. tab-item:: C++
         :sync: cpp
   
         .. code-block:: console
      
            sync_benchmark googlenet-v1.xml


Sample Output
####################


.. tab-set::

   .. tab-item:: Python
      :sync: python

      The application outputs performance results.
      
      .. code-block:: console
      
         [ INFO ] OpenVINO:
         [ INFO ] Build ................................. <version>
         [ INFO ] Count:          2333 iterations
         [ INFO ] Duration:       10003.59 ms
         [ INFO ] Latency:
         [ INFO ]     Median:     3.90 ms
         [ INFO ]     Average:    4.29 ms
         [ INFO ]     Min:        3.30 ms
         [ INFO ]     Max:        10.11 ms
         [ INFO ] Throughput: 233.22 FPS

   .. tab-item:: C++
      :sync: cpp

      The application outputs performance results.
      
      .. code-block:: console
      
         [ INFO ] OpenVINO:
         [ INFO ] Build ................................. <version>
         [ INFO ] Count:      992 iterations
         [ INFO ] Duration:   15009.8 ms
         [ INFO ] Latency:
         [ INFO ]        Median:     14.00 ms
         [ INFO ]        Average:    15.13 ms
         [ INFO ]        Min:        9.33 ms
         [ INFO ]        Max:        53.60 ms
         [ INFO ] Throughput: 66.09 FPS


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
- :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
