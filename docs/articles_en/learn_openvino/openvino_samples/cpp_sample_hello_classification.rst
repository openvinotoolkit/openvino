.. {#openvino_inference_engine_samples_hello_classification_README}

Hello Classification C++ Sample
===============================


.. meta::
   :description: Learn how to do inference of image 
                 classification models using Synchronous Inference Request 
                 (C++) API.


This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API. 

Models with only one input and output are supported.

.. tab-set::

   .. tab-item:: Requirements 

      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Options                             | Values                                                                                                                                                                                |
      +=====================================+=======================================================================================================================================================================================+
      | Validated Models                    | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`                                                                                        |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Format                        | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                                       |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Supported devices                   | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                                                                                                                  |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Other language realization          | :doc:`C <openvino_inference_engine_ie_bridges_c_samples_hello_classification_README>`, :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>` |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API

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

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/cpp/hello_classification/main.cpp 
         :language: cpp


How It Works
############

At startup, the sample application reads command line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin and performs synchronous inference. Then processes output data and write it to a standard output stream.

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

.. code-block:: console
   
   hello_classification <path_to_model> <path_to_image> <device_name>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note::
  
   - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using ``mo`` with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
+++++++

1. Install the ``openvino-dev`` Python package to use Open Model Zoo Tools:
   
   .. code-block:: console
      
      python -m pip install openvino-dev[caffe]

2. Download a pre-trained model using:
   
   .. code-block:: console
      
      omz_downloader --name googlenet-v1

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:
   
   .. code-block:: console
      
      omz_converter --name googlenet-v1

4. Perform inference of ``car.bmp`` using the ``googlenet-v1`` model on a ``GPU``, for example:
   
   .. code-block:: console
      
      hello_classification googlenet-v1.xml car.bmp GPU

Sample Output
#############

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

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`


