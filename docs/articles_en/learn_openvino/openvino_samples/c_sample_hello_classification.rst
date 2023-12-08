.. {#openvino_inference_engine_ie_bridges_c_samples_hello_classification_README}

Hello Classification C Sample
=============================


.. meta::
   :description: Learn how to do inference of image 
                 classification models, such as alexnet and googlenet-v1, using 
                 Synchronous Inference Request (C) API.


This sample demonstrates how to execute an inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API and input auto-resize feature.

.. tab-set::

   .. tab-item:: Requirements 

      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Options                    | Values                                                                                                                                                                     |
      +============================+============================================================================================================================================================================+
      | Validated Models           | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`                                                                             |
      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Format               | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                             |
      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Validated images           | The sample uses OpenCV\* to `read input image <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`__ (\*.bmp, \*.png)         |
      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Supported devices          | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                                                                                                       |
      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Other language realization | :doc:`C++ <openvino_inference_engine_samples_hello_classification_README>`, :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>` |
      +----------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   .. tab-item:: C API 

      Hello Classification C sample application demonstrates how to use the C API from OpenVINO in applications.

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

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/c/hello_classification/main.c 
         :language: c

How It Works
############

Upon the start-up, the sample application reads command line parameters, loads specified network and an image to the Inference Engine plugin.
Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream.

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in Inference Engine Samples guide.

Running
#######

To run the sample, you need specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note:: 
  
   - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using ``mo`` with `reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

Example
+++++++

1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader):
   
   .. code-block:: console
      
      python <path_to_omz_tools>/downloader.py --name alexnet

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:
   
   .. code-block:: console
      
      python <path_to_omz_tools>/converter.py --name alexnet

3. Perform inference of ``car.bmp`` using ``alexnet`` model on a ``GPU``, for example:
   
   .. code-block:: console
      
      <path_to_sample>/hello_classification_c <path_to_model>/alexnet.xml <path_to_image>/car.bmp GPU

Sample Output
#############

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
   
   This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

See Also
########

- :doc:`Integrate OpenVINO™ into Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
- :doc:`C API Reference <pot_compression_api_README>`


