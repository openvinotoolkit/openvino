.. {#openvino_sample_hello_classification}

Hello Classification Sample
===========================


.. meta::
   :description: Learn how to do inference of image classification 
                 models using Synchronous Inference Request API (Python, C++, C).


This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API. 

Models with only one input and output are supported.

Requirements
####################

+--------------------+------------------------------------------------------------------------------------------------+
| Options            | Values                                                                                         |
+====================+================================================================================================+
| Validated Models   | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>` |
+--------------------+------------------------------------------------------------------------------------------------+
| Model Format       | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                      |
+--------------------+------------------------------------------------------------------------------------------------+
| Supported devices  | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                           |
+--------------------+------------------------------------------------------------------------------------------------+


How It Works
####################

At startup, the sample application reads command-line parameters, prepares input data, 
loads a specified model and image to the OpenVINO™ Runtime plugin (Inference Engine in C API), 
performs synchronous inference, and processes output data, logging each step in a standard output stream.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/python/hello_classification/hello_classification.py
                  :language: python

         .. tab-item:: API
      
            The following Python API is used in the application:
      
            +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Feature                     | API                                                                                                                                                                                                                                       | Description                                                                                                                                                                                |
            +=============================+===========================================================================================================================================================================================================================================+============================================================================================================================================================================================+
            | Basic Infer Flow            | `openvino.runtime.Core <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html>`__ ,                                                                                                                   |                                                                                                                                                                                            |
            |                             | `openvino.runtime.Core.read_model <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.read_model>`__ ,                                                                       |                                                                                                                                                                                            |
            |                             | `openvino.runtime.Core.compile_model <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.compile_model>`__                                                                   | Common API to do inference                                                                                                                                                                 |
            +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Synchronous Infer           | `openvino.runtime.CompiledModel.infer_new_request <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.infer_new_request>`__                                | Do synchronous inference                                                                                                                                                                   |
            +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Model Operations            | `openvino.runtime.Model.inputs <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.inputs>`__ ,                                                                            | Managing of model                                                                                                                                                                          |
            |                             | `openvino.runtime.Model.outputs <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.outputs>`__                                                                            |                                                                                                                                                                                            |
            +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Preprocessing               | `openvino.preprocess.PrePostProcessor <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html>`__ ,                                                                                     | Set image of the original size as input for a model with other input size. Resize and layout conversions will be performed automatically by the corresponding plugin just before inference |
            |                             | `openvino.preprocess.InputTensorInfo.set_element_type <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_element_type>`__ ,                 |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.InputTensorInfo.set_layout <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_layout>`__ ,                             |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.InputTensorInfo.set_spatial_static_shape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_spatial_static_shape>`__ , |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.PreProcessSteps.resize <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html#openvino.preprocess.PreProcessSteps.resize>`__ ,                                     |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.InputModelInfo.set_layout <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.InputModelInfo.html#openvino.preprocess.InputModelInfo.set_layout>`__ ,                                |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.OutputTensorInfo.set_element_type <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.OutputTensorInfo.html#openvino.preprocess.OutputTensorInfo.set_element_type>`__ ,              |                                                                                                                                                                                            |
            |                             | `openvino.preprocess.PrePostProcessor.build <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html#openvino.preprocess.PrePostProcessor.build>`__                                      |                                                                                                                                                                                            |
            +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/cpp/hello_classification/main.cpp
                  :language: cpp

         .. tab-item:: API
      
            The following C++ API is used in the application:
      
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Feature                             | API                                                            | Description                                                                                                                                                                             |
            +=====================================+================================================================+=========================================================================================================================================================================================+
            | OpenVINO Runtime Version            | ``ov::get_openvino_version``                                   | Get Openvino API version                                                                                                                                                                |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Basic Infer Flow                    | ``ov::Core::read_model``,                                      | Common API to do inference: read and compile a model, create an infer request, configure input and output tensors                                                                       |
            |                                     | ``ov::Core::compile_model``,                                   |                                                                                                                                                                                         |
            |                                     | ``ov::CompiledModel::create_infer_request``,                   |                                                                                                                                                                                         |
            |                                     | ``ov::InferRequest::set_input_tensor``,                        |                                                                                                                                                                                         |
            |                                     | ``ov::InferRequest::get_output_tensor``                        |                                                                                                                                                                                         |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Synchronous Infer                   | ``ov::InferRequest::infer``                                    | Do synchronous inference                                                                                                                                                                |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Model Operations                    | ``ov::Model::inputs``,                                         | Get inputs and outputs of a model                                                                                                                                                       |
            |                                     | ``ov::Model::outputs``                                         |                                                                                                                                                                                         |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Tensor Operations                   | ``ov::Tensor::get_shape``                                      | Get a tensor shape                                                                                                                                                                      |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Preprocessing                       | ``ov::preprocess::InputTensorInfo::set_element_type``,         | Set image of the original size as input for a model with other input size. Resize and layout conversions are performed automatically by the corresponding plugin just before inference. |
            |                                     | ``ov::preprocess::InputTensorInfo::set_layout``,               |                                                                                                                                                                                         |
            |                                     | ``ov::preprocess::InputTensorInfo::set_spatial_static_shape``, |                                                                                                                                                                                         |
            |                                     | ``ov::preprocess::PreProcessSteps::resize``,                   |                                                                                                                                                                                         |
            |                                     | ``ov::preprocess::InputModelInfo::set_layout``,                |                                                                                                                                                                                         |
            |                                     | ``ov::preprocess::OutputTensorInfo::set_element_type``,        |                                                                                                                                                                                         |
            |                                     | ``ov::preprocess::PrePostProcessor::build``                    |                                                                                                                                                                                         |
            +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   .. tab-item:: C
      :sync: c

      .. tab-set::
      
         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/c/hello_classification/main.c 
                  :language: c

         .. tab-item:: API 
      
            The following C API is used in the application:
      
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Feature                             | API                                                         | Description                                                                                                                                                                             |
            +=====================================+=============================================================+=========================================================================================================================================================================================+
            | OpenVINO Runtime Version            | ``ov_get_openvino_version``                                 | Get Openvino API version                                                                                                                                                                |
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Basic Infer Flow                    | ``ov_core_create``,                                         | Common API to do inference: read and compile a model, create an infer request, configure input and output tensors                                                                       |
            |                                     | ``ov_core_read_model``,                                     |                                                                                                                                                                                         |
            |                                     | ``ov_core_compile_model``,                                  |                                                                                                                                                                                         |
            |                                     | ``ov_compiled_model_create_infer_request``,                 |                                                                                                                                                                                         |
            |                                     | ``ov_infer_request_set_input_tensor_by_index``,             |                                                                                                                                                                                         |
            |                                     | ``ov_infer_request_get_output_tensor_by_index``             |                                                                                                                                                                                         |
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Synchronous Infer                   | ``ov_infer_request_infer``                                  | Do synchronous inference                                                                                                                                                                |
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Model Operations                    | ``ov_model_const_input``,                                   | Get inputs and outputs of a model                                                                                                                                                       |
            |                                     | ``ov_model_const_output``                                   |                                                                                                                                                                                         +
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Tensor Operations                   | ``ov_tensor_create_from_host_ptr``                          | Create a tensor shape                                                                                                                                                                   |
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            | Preprocessing                       | ``ov_preprocess_prepostprocessor_create``,                  | Set image of the original size as input for a model with other input size. Resize and layout conversions are performed automatically by the corresponding plugin just before inference. |
            |                                     | ``ov_preprocess_prepostprocessor_get_input_info_by_index``, |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_input_info_get_tensor_info``,               |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_input_tensor_info_set_from``,               |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_input_tensor_info_set_layout``,             |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_input_info_get_preprocess_steps``,          |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_preprocess_steps_resize``,                  |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_input_model_info_set_layout``,              |                                                                                                                                                                                         |
            |                                     | ``ov_preprocess_output_set_element_type``,                  |                                                                                                                                                                                         | 
            |                                     | ``ov_preprocess_prepostprocessor_build``                    |                                                                                                                                                                                         |
            +-------------------------------------+-------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


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

         python hello_classification.py <path_to_model> <path_to_image> <device_name>

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         hello_classification <path_to_model> <path_to_image> <device_name>

   .. tab-item:: C
      :sync: c

      .. code-block:: console

         hello_classification_c <path_to_model> <path_to_image> <device_name>

To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model 
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at 
  `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note::

   - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR 
     channels order. If you trained your model to work with RGB order, you need 
     to manually rearrange the default channels order in the sample or demo 
     application or reconvert your model using model conversion API with 
     ``reverse_input_channels`` argument specified. For more information about 
     the argument, refer to **When to Reverse Input Channels** section of 
     :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is 
     converted to the intermediate representation (IR) format (\*.xml + \*.bin) 
     using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
++++++++++++++++++++

1. Download a pre-trained model.
2. If a model is not in the IR or ONNX format, it must be converted by using:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            import openvino as ov

            ov_model = ov.convert_model('./models/alexnet')
            # or, when model is a Python model object
            ov_model = ov.convert_model(alexnet)

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: console

            ovc ./models/alexnet

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            mo --input_model ./models/googlenet-v1

      .. tab-item:: C
         :sync: c

         .. code-block:: console

            mo --input_model ./models/alexnet


3. Perform inference of an image, using a model on a ``GPU``, for example:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: console

            python hello_classification.py ./models/alexnet/alexnet.xml ./images/banana.jpg GPU

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            hello_classification ./models/googlenet-v1.xml ./images/car.bmp GPU

      .. tab-item:: C
         :sync: c

         .. code-block:: console

            hello_classification_c alexnet.xml ./opt/intel/openvino/samples/scripts/car.png GPU

Sample Output
#############

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The sample application logs each step in a standard output stream and 
      outputs top-10 inference results.

      .. code-block:: console

         [ INFO ] Creating OpenVINO Runtime Core
         [ INFO ] Reading the model: /models/alexnet/alexnet.xml
         [ INFO ] Loading the model to the plugin
         [ INFO ] Starting inference in synchronous mode
         [ INFO ] Image path: /images/banana.jpg
         [ INFO ] Top 10 results:     
         [ INFO ] class_id probability
         [ INFO ] --------------------
         [ INFO ] 954      0.9703885
         [ INFO ] 666      0.0219518
         [ INFO ] 659      0.0033120
         [ INFO ] 435      0.0008246
         [ INFO ] 809      0.0004433
         [ INFO ] 502      0.0003852
         [ INFO ] 618      0.0002906
         [ INFO ] 910      0.0002848
         [ INFO ] 951      0.0002427
         [ INFO ] 961      0.0002213
         [ INFO ]
         [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

   .. tab-item:: C++
      :sync: cpp

      The application outputs top-10 inference results.

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>
         [ INFO ]
         [ INFO ] Loading model files: /models/googlenet-v1.xml
         [ INFO ] model name: GoogleNet
         [ INFO ]     inputs
         [ INFO ]         input name: data
         [ INFO ]         input type: f32
         [ INFO ]         input shape: {1, 3, 224, 224}
         [ INFO ]     outputs
         [ INFO ]         output name: prob
         [ INFO ]         output type: f32
         [ INFO ]         output shape: {1, 1000}

         Top 10 results:

         Image /images/car.bmp

         classid probability
         ------- -----------
         656     0.8139648
         654     0.0550537
         468     0.0178375
         436     0.0165405
         705     0.0111694
         817     0.0105820
         581     0.0086823
         575     0.0077515
         734     0.0064468
         785     0.0043983

   .. tab-item:: C
      :sync: c

      The application outputs top-10 inference results.

      .. code-block:: console

         Top 10 results:

         Image /opt/intel/openvino/samples/scripts/car.png

         classid probability
         ------- -----------
         656       0.666479
         654       0.112940
         581       0.068487
         874       0.033385
         436       0.026132
         817       0.016731
         675       0.010980
         511       0.010592
         569       0.008178
         717       0.006336

         This sample is an API example, for any performance measurements use the dedicated benchmark_app tool.


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
- :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
- :doc:`C API Reference <pot_compression_api_README>`
