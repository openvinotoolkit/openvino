# Hello NV12 Input Classification C++ Sample {#openvino_inference_engine_samples_hello_nv12_input_classification_README}

@sphinxdirective

.. meta::
   :description: Learn how to do inference of image 
                 classification models with images in NV12 color format using  
                 Synchronous Inference Request (C++) API.


This sample demonstrates how to execute an inference of image classification models with images in NV12 color format using Synchronous Inference Request API.

.. tab-set::

   .. tab-item:: Requirements 

      +-------------------------------------+--------------------------------------------------------------------------------------------------+
      | Options                             | Values                                                                                           |
      +=====================================+==================================================================================================+
      | Validated Models                    | :doc:`alexnet <omz_models_model_alexnet>`                                                        |
      +-------------------------------------+--------------------------------------------------------------------------------------------------+
      | Model Format                        | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                  |
      +-------------------------------------+--------------------------------------------------------------------------------------------------+
      | Validated images                    | An uncompressed image in the NV12 color format - \*.yuv                                          |
      +-------------------------------------+--------------------------------------------------------------------------------------------------+
      | Supported devices                   | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                             |
      +-------------------------------------+--------------------------------------------------------------------------------------------------+
      | Other language realization          | :doc:`C <openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README>` |
      +-------------------------------------+--------------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API 

      The following C++ API is used in the application:

      +-------------------------------------+-------------------------------------------------------------+-------------------------------------------+
      | Feature                             | API                                                         | Description                               |
      +=====================================+=============================================================+===========================================+
      | Node Operations                     | ``ov::Output::get_any_name``                                | Get a layer name                          |
      +-------------------------------------+-------------------------------------------------------------+-------------------------------------------+
      | Infer Request Operations            | ``ov::InferRequest::set_tensor``,                           | Operate with tensors                      |
      |                                     | ``ov::InferRequest::get_tensor``                            |                                           |
      +-------------------------------------+-------------------------------------------------------------+-------------------------------------------+
      | Preprocessing                       | ``ov::preprocess::InputTensorInfo::set_color_format``,      | Change the color format of the input data |
      |                                     | ``ov::preprocess::PreProcessSteps::convert_element_type``,  |                                           |
      |                                     | ``ov::preprocess::PreProcessSteps::convert_color``          |                                           |
      +-------------------------------------+-------------------------------------------------------------+-------------------------------------------+


      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_inference_engine_samples_hello_classification_README>`.

   .. tab-item:: Sample Code
   
      .. doxygensnippet:: samples/cpp/hello_nv12_input_classification/main.cpp  
         :language: cpp

How It Works
############

At startup, the sample application reads command line parameters, loads the specified model and an image in the NV12 color format to an OpenVINO™ Runtime plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application outputs data to the standard output stream. You can place labels in .labels file near the model to get pretty output.

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

.. code-block:: console
   
   hello_nv12_input_classification <path_to_model> <path_to_image> <image_size> <device_name>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

The sample accepts an uncompressed image in the NV12 color format. To run the sample, you need to convert your BGR/RGB image to NV12. To do this, you can use one of the widely available tools such as FFmpeg\* or GStreamer\*. The following command shows how to convert an ordinary image into an uncompressed NV12 image using FFmpeg:

.. code-block:: sh
   
   ffmpeg -i cat.jpg -pix_fmt nv12 car.yuv


.. note::
  
   - Because the sample reads raw image files, you should provide a correct image size along with the image path. The sample expects the logical size of the image, not the buffer size. For example, for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the buffer size is 640x720.
   - By default, this sample expects that model input has BGR channels order. If you trained your model to work with RGB order, you need to reconvert your model using ``mo`` with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
+++++++

1. Install openvino-dev python package if you don't have it to use Open Model Zoo Tools:
   
   .. code-block:: console
      
      python -m pip install openvino-dev[caffe]

2. Download a pre-trained model:

   .. code-block:: console
      
      omz_downloader --name alexnet

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:
   
   .. code-block:: console
      
      omz_converter --name alexnet

4. Perform inference of NV12 image using ``alexnet`` model on a ``CPU``, for example:
   
   .. code-block:: console
      
      hello_nv12_input_classification alexnet.xml car.yuv 300x300 CPU


Sample Output
#############

The application outputs top-10 inference results.

.. code-block:: console
   
   [ INFO ] OpenVINO Runtime version ......... <version>
   [ INFO ] Build ........... <build>
   [ INFO ]
   [ INFO ] Loading model files: \models\alexnet.xml
   [ INFO ] model name: AlexNet
   [ INFO ]     inputs
   [ INFO ]         input name: data
   [ INFO ]         input type: f32
   [ INFO ]         input shape: {1, 3, 227, 227}
   [ INFO ]     outputs
   [ INFO ]         output name: prob
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {1, 1000}
   
   Top 10 results:
   
   Image \images\car.yuv
   
   classid probability
   ------- -----------
   656     0.6668988
   654     0.1125269
   581     0.0679280
   874     0.0340229
   436     0.0257744
   817     0.0169367
   675     0.0110199
   511     0.0106134
   569     0.0083373
   717     0.0061734


See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective

