.. {#openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README}

Hello NV12 Input Classification C Sample
========================================


.. meta::
   :description: Learn how to do inference of an image 
                 classification model with images in NV12 color format using  
                 Synchronous Inference Request (C) API.


This sample demonstrates how to execute an inference of image classification networks like AlexNet with images in NV12 color format using Synchronous Inference Request API.

Hello NV12 Input Classification C Sample demonstrates how to use the NV12 automatic input pre-processing API in your applications.

.. tab-set::

   .. tab-item:: Requirements 

      +-----------------------------------------+---------------------------------------------------------------------------------------+
      | Options                                 | Values                                                                                |
      +=========================================+=======================================================================================+
      | Validated Models                        | :doc:`alexnet <omz_models_model_alexnet>`                                             |
      +-----------------------------------------+---------------------------------------------------------------------------------------+
      | Model Format                            | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)        |
      +-----------------------------------------+---------------------------------------------------------------------------------------+
      | Validated images                        | An uncompressed image in the NV12 color format - \*.yuv                               |
      +-----------------------------------------+---------------------------------------------------------------------------------------+
      | Supported devices                       | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                  |
      +-----------------------------------------+---------------------------------------------------------------------------------------+
      | Other language realization              | :doc:`C++ <openvino_inference_engine_samples_hello_nv12_input_classification_README>` |
      +-----------------------------------------+---------------------------------------------------------------------------------------+

   .. tab-item:: C API 

      +-----------------------------------------+-----------------------------------------------------------+--------------------------------------------------------+
      | Feature                                 | API                                                       | Description                                            |
      +=========================================+===========================================================+========================================================+
      | Node Operations                         | ``ov_port_get_any_name``                                  | Get a layer name                                       |
      +-----------------------------------------+-----------------------------------------------------------+--------------------------------------------------------+
      | Infer Request Operations                | ``ov_infer_request_set_tensor``,                          | Operate with tensors                                   |
      |                                         | ``ov_infer_request_get_output_tensor_by_index``           |                                                        |
      +-----------------------------------------+-----------------------------------------------------------+--------------------------------------------------------+
      | Preprocessing                           | ``ov_preprocess_input_tensor_info_set_color_format``,     | Change the color format of the input data              |
      |                                         | ``ov_preprocess_preprocess_steps_convert_element_type``,  |                                                        |
      |                                         | ``ov_preprocess_preprocess_steps_convert_color``          |                                                        |
      +-----------------------------------------+-----------------------------------------------------------+--------------------------------------------------------+


      Basic Inference Engine API is covered by :doc:`Hello Classification C sample <openvino_inference_engine_ie_bridges_c_samples_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/c/hello_nv12_input_classification/main.c
         :language: c

How It Works
############

Upon the start-up, the sample application reads command-line parameters, loads specified network and an image in the NV12 color format to an Inference Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream.

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in Inference Engine Samples guide.

Running
#######

To run the sample, you need specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

The sample accepts an uncompressed image in the NV12 color format. To run the sample, you need to convert your BGR/RGB image to NV12. To do this, you can use one of the widely available tools such as FFmpeg\* or GStreamer\*. The following command shows how to convert an ordinary image into an uncompressed NV12 image using FFmpeg:

.. code-block:: sh
   
   ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv

.. note::
  
   - Because the sample reads raw image files, you should provide a correct image size along with the image path. The sample expects the logical size of the image, not the buffer size. For example, for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the buffer size is 640x720.
   - By default, this sample expects that network input has BGR channels order. If you trained your model to work with RGB order, you need to reconvert your model using ``mo`` with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
+++++++

1. Download a pre-trained model using :doc:`Model Downloader <omz_tools_downloader>`:
   
   .. code-block:: console
      
      python <path_to_omz_tools>/downloader.py --name alexnet

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:
   
   .. code-block:: console

      python <path_to_omz_tools>/converter.py --name alexnet

3. Perform inference of NV12 image using `alexnet` model on a `CPU`, for example:
   
   .. code-block:: console
      
      <path_to_sample>/hello_nv12_input_classification_c <path_to_model>/alexnet.xml <path_to_image>/cat.yuv 300x300 CPU

Sample Output
#############

The application outputs top-10 inference results.

.. code-block:: console
   
   Top 10 results:
   
   Image ./cat.yuv
   
   classid probability
   ------- -----------
   435       0.091733
   876       0.081725
   999       0.069305
   587       0.043726
   666       0.038957
   419       0.032892
   285       0.030309
   700       0.029941
   696       0.021628
   855       0.020339
   
   This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

See Also
########

- :doc:`Integrate the OpenVINO™ into Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
- `C API Reference <https://docs.openvino.ai/2023.2/api/api_reference.html>`__


