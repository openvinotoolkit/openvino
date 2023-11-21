# Hello Reshape SSD Python Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README}

@sphinxdirective

.. meta::
   :description: Learn how to do inference of object detection 
                 models using shape inference feature and Synchronous 
                 Inference Request (Python) API.


This sample demonstrates how to do synchronous inference of object detection models using :doc:`Shape Inference feature <openvino_docs_OV_UG_ShapeInference>`.  

Models with only 1 input and output are supported.

.. tab-set::

   .. tab-item:: Requirements 

      +------------------------------------+---------------------------------------------------------------------------+
      | Options                            | Values                                                                    |
      +====================================+===========================================================================+
      | Validated Models                   | :doc:`mobilenet-ssd <omz_models_model_mobilenet_ssd>`                     |
      +------------------------------------+---------------------------------------------------------------------------+
      | Validated Layout                   | NCHW                                                                      |
      +------------------------------------+---------------------------------------------------------------------------+
      | Model Format                       | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx) |
      +------------------------------------+---------------------------------------------------------------------------+
      | Supported devices                  | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`      |
      +------------------------------------+---------------------------------------------------------------------------+
      | Other language realization         | :doc:`C++ <openvino_inference_engine_samples_hello_reshape_ssd_README>`   |
      +------------------------------------+---------------------------------------------------------------------------+

   .. tab-item:: Python API 

      The following Python API is used in the application:

      +------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------+
      | Feature                            | API                                                                                                                                                                            | Description                          |
      +====================================+================================================================================================================================================================================+======================================+
      | Model Operations                   | `openvino.runtime.Model.reshape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape>`__ ,               | Managing of model                    |
      |                                    | `openvino.runtime.Model.input <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input>`__ ,                   |                                      |
      |                                    | `openvino.runtime.Output.get_any_name <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Output.html#openvino.runtime.Output.get_any_name>`__ ,  |                                      |
      |                                    | `openvino.runtime.PartialShape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.PartialShape.html>`__                                          |                                      |
      +------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------+

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification Python* Sample <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/python/hello_reshape_ssd/hello_reshape_ssd.py  
         :language: python


How It Works
############

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image, logging each step in a standard output stream.

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
#######

.. code-block:: console
   
   python hello_reshape_ssd.py <path_to_model> <path_to_image> <device_name>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`.

.. note::
  
   - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using model conversion API with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
+++++++

1. Install the ``openvino-dev`` Python package to use Open Model Zoo Tools:
   
   .. code-block:: console
      
      python -m pip install openvino-dev[caffe]

2. Download a pre-trained model:
   
   .. code-block:: console
      
      omz_downloader --name mobilenet-ssd

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:
   
   .. code-block:: console
      
      omz_converter --name mobilenet-ssd

4. Perform inference of ``banana.jpg`` using ``ssdlite_mobilenet_v2`` model on a ``GPU``, for example:
   
   .. code-block:: console
      
      python hello_reshape_ssd.py mobilenet-ssd.xml banana.jpg GPU

Sample Output
#############

The sample application logs each step in a standard output stream and creates an output image, drawing bounding boxes for inference results with an over 50% confidence.

.. code-block:: console
   
   [ INFO ] Creating OpenVINO Runtime Core
   [ INFO ] Reading the model: C:/test_data/models/mobilenet-ssd.xml
   [ INFO ] Reshaping the model to the height and width of the input image
   [ INFO ] Loading the model to the plugin
   [ INFO ] Starting inference in synchronous mode
   [ INFO ] Found: class_id = 52, confidence = 0.98, coords = (21, 98), (276, 210)
   [ INFO ] Image out.bmp was created!
   [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective

