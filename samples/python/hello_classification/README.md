# Hello Classification Python Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_classification_README}

@sphinxdirective

.. meta::
   :description: Learn how to do inference of image classification 
                 models using Synchronous Inference Request (Python) API.


This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API. 

Models with only 1 input and output are supported.

.. tab-set::

   .. tab-item:: Requirements 

      +-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Options                           | Values                                                                                                                                                            |
      +===================================+===================================================================================================================================================================+
      | Validated Models                  | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`                                                                    |
      +-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Format                      | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)                                                                                         |
      +-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Supported devices                 | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                                                                                              |
      +-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Other language realization        | :doc:`C++ <openvino_inference_engine_samples_hello_classification_README>`, :doc:`C <openvino_inference_engine_ie_bridges_c_samples_hello_classification_README>` |
      +-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   
   .. tab-item:: Python API

      The following Python API is used in the application:

      +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Feature                     | API                                                                                                                                                                                                                                       | Description                                                                                                                                                                                |
      +=============================+===========================================================================================================================================================================================================================================+============================================================================================================================================================================================+
      | Basic Infer Flow            | `openvino.runtime.Core <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.Core.html>`__ ,                                                                                                                   |                                                                                                                                                                                            |
      |                             | `openvino.runtime.Core.read_model <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.read_model>`__ ,                                                                       |                                                                                                                                                                                            |
      |                             | `openvino.runtime.Core.compile_model <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.compile_model>`__                                                                   | Common API to do inference                                                                                                                                                                 |
      +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Synchronous Infer           | `openvino.runtime.CompiledModel.infer_new_request <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.infer_new_request>`__                                | Do synchronous inference                                                                                                                                                                   |
      +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Operations            | `openvino.runtime.Model.inputs <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.inputs>`__ ,                                                                            | Managing of model                                                                                                                                                                          |
      |                             | `openvino.runtime.Model.outputs <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.outputs>`__                                                                            |                                                                                                                                                                                            |
      +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Preprocessing               | `openvino.preprocess.PrePostProcessor <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html>`__ ,                                                                                     | Set image of the original size as input for a model with other input size. Resize and layout conversions will be performed automatically by the corresponding plugin just before inference |
      |                             | `openvino.preprocess.InputTensorInfo.set_element_type <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_element_type>`__ ,                 |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.InputTensorInfo.set_layout <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_layout>`__ ,                             |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.InputTensorInfo.set_spatial_static_shape <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.InputTensorInfo.html#openvino.preprocess.InputTensorInfo.set_spatial_static_shape>`__ , |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.PreProcessSteps.resize <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html#openvino.preprocess.PreProcessSteps.resize>`__ ,                                     |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.InputModelInfo.set_layout <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.InputModelInfo.html#openvino.preprocess.InputModelInfo.set_layout>`__ ,                                |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.OutputTensorInfo.set_element_type <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.OutputTensorInfo.html#openvino.preprocess.OutputTensorInfo.set_element_type>`__ ,              |                                                                                                                                                                                            |
      |                             | `openvino.preprocess.PrePostProcessor.build <https://docs.openvino.ai/2023.1/api/ie_python_api/_autosummary/openvino.preprocess.PrePostProcessor.html#openvino.preprocess.PrePostProcessor.build>`__                                      |                                                                                                                                                                                            |
      +-----------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/python/hello_classification/hello_classification.py
         :language: python

How It Works
############

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous inference, and processes output data, logging each step in a standard output stream.

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
#######

.. code-block:: console
   
   python hello_classification.py <path_to_model> <path_to_image> <device_name>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available at `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note::
  
   - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using model conversion API with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
+++++++

1. Install the ``openvino-dev`` Python package to use Open Model Zoo Tools:
   
   .. code-block:: console
      
      python -m pip install openvino-dev[caffe]

2. Download a pre-trained model:
   
   .. code-block:: console
      
      omz_downloader --name alexnet

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:
   
   .. code-block:: console
      
      omz_converter --name alexnet

4. Perform inference of ``banana.jpg`` using the ``alexnet`` model on a ``GPU``, for example:
   
   .. code-block:: console
      
      python hello_classification.py alexnet.xml banana.jpg GPU

Sample Output
#############

The sample application logs each step in a standard output stream and outputs top-10 inference results.

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

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective

