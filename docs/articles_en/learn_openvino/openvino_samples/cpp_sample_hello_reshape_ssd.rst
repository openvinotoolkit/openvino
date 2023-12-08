.. {#openvino_inference_engine_samples_hello_reshape_ssd_README}

Hello Reshape SSD C++ Sample
============================


.. meta::
   :description: Learn how to do inference of object 
                 detection models using shape inference feature and Synchronous 
                 Inference Request (C++) API.


This sample demonstrates how to do synchronous inference of object detection models using :doc:`input reshape feature <openvino_docs_OV_UG_ShapeInference>`.
Models with only one input and output are supported.

.. tab-set::

   .. tab-item:: Requirements 

      +----------------------------------+---------------------------------------------------------------------------------------------+
      | Options                          | Values                                                                                      |
      +==================================+=============================================================================================+
      | Validated Models                 | :doc:`person-detection-retail-0013 <omz_models_model_person_detection_retail_0013>`         |
      +----------------------------------+---------------------------------------------------------------------------------------------+
      | Model Format                     | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)             |
      +----------------------------------+---------------------------------------------------------------------------------------------+
      | Supported devices                | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                        |
      +----------------------------------+---------------------------------------------------------------------------------------------+
      | Other language realization       | :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README>` |
      +----------------------------------+---------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API 

      The following C++ API is used in the application:

      +----------------------------------+-------------------------------------------------------------+------------------------------------------------+
      | Feature                          | API                                                         | Description                                    |
      +==================================+=============================================================+================================================+
      | Node operations                  | ``ov::Node::get_type_info``,                                | Get a node info                                |
      |                                  | ``ngraph::op::DetectionOutput::get_type_info_static``,      |                                                |
      |                                  | ``ov::Output::get_any_name``,                               |                                                |
      |                                  | ``ov::Output::get_shape``                                   |                                                |
      +----------------------------------+-------------------------------------------------------------+------------------------------------------------+
      | Model Operations                 | ``ov::Model::get_ops``,                                     | Get model nodes, reshape input                 |
      |                                  | ``ov::Model::reshape``                                      |                                                |
      +----------------------------------+-------------------------------------------------------------+------------------------------------------------+
      | Tensor Operations                | ``ov::Tensor::data``                                        | Get a tensor data                              |
      +----------------------------------+-------------------------------------------------------------+------------------------------------------------+
      | Preprocessing                    | ``ov::preprocess::PreProcessSteps::convert_element_type``,  | Model input preprocessing                      |
      |                                  | ``ov::preprocess::PreProcessSteps::convert_layout``         |                                                |
      +----------------------------------+-------------------------------------------------------------+------------------------------------------------+

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_inference_engine_samples_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/cpp/hello_reshape_ssd/main.cpp 
         :language: cpp


How It Works
############

Upon the start-up the sample application reads command line parameters, loads specified network and image to the Inference
Engine plugin. Then, the sample creates an synchronous inference request object. When inference is done, the application creates output image and output data to the standard output stream.

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

.. code-block:: console
   
   hello_reshape_ssd <path_to_model> <path_to_image> <device>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note::
  
   - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using ``mo`` with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

Example
+++++++

1. Install openvino-dev python package if you don't have it to use Open Model Zoo Tools:

   .. code-block:: console
      
      python -m pip install openvino-dev

2. Download a pre-trained model using:

   .. code-block:: console
      
      omz_downloader --name person-detection-retail-0013

3. ``person-detection-retail-0013`` does not need to be converted, because it is already in necessary format, so you can skip this step. If you want to use another model that is not in the IR or ONNX format, you can convert it using the model converter script:

   .. code-block:: console
      
      omz_converter --name <model_name>

4. Perform inference of ``person_detection.bmp`` using ``person-detection-retail-0013`` model on a ``GPU``, for example:
   
   .. code-block:: console
      
      hello_reshape_ssd person-detection-retail-0013.xml person_detection.bmp GPU

Sample Output
#############

The application renders an image with detected objects enclosed in rectangles. It outputs the list of classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.

.. code-block:: console
   
   [ INFO ] OpenVINO Runtime version ......... <version>
   [ INFO ] Build ........... <build>
   [ INFO ]
   [ INFO ] Loading model files: \models\person-detection-retail-0013.xml
   [ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
   [ INFO ]     inputs
   [ INFO ]         input name: data
   [ INFO ]         input type: f32
   [ INFO ]         input shape: {1, 3, 320, 544}
   [ INFO ]     outputs
   [ INFO ]         output name: detection_out
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {1, 1, 200, 7}
   Reshape network to the image size = [960x1699]
   [ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
   [ INFO ]     inputs
   [ INFO ]         input name: data
   [ INFO ]         input type: f32
   [ INFO ]         input shape: {1, 3, 960, 1699}
   [ INFO ]     outputs
   [ INFO ]         output name: detection_out
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {1, 1, 200, 7}
   [0,1] element, prob = 0.716309,    (852,187)-(983,520)
   The resulting image was saved in the file: hello_reshape_ssd_output.bmp
   
   This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`


