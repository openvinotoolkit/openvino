# Image Classification Async C++ Sample {#openvino_inference_engine_samples_classification_sample_async_README}

@sphinxdirective

.. meta::
   :description: Learn how to do inference of image 
                 classification models using Asynchronous Inference Request 
                 (C++) API.


This sample demonstrates how to do inference of image classification models using Asynchronous Inference Request API. 
 
Models with only one input and output are supported.

In addition to regular images, the sample also supports single-channel ``ubyte`` images as an input for LeNet model.

.. tab-set::

   .. tab-item:: Requirements 

      +----------------------------+-------------------------------------------------------------------------------------------------------+
      | Options                    | Values                                                                                                |
      +============================+=======================================================================================================+
      | Validated Models           | :doc:`alexnet <omz_models_model_alexnet>`, :doc:`googlenet-v1 <omz_models_model_googlenet_v1>`        |
      +----------------------------+-------------------------------------------------------------------------------------------------------+
      | Model Format               | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                       |
      +----------------------------+-------------------------------------------------------------------------------------------------------+
      | Supported devices          | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                                  |
      +----------------------------+-------------------------------------------------------------------------------------------------------+
      | Other language realization | :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README>` |
      +----------------------------+-------------------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API

      The following C++ API is used in the application:

      +--------------------------+-----------------------------------------------------------------------+----------------------------------------------------------------------------------------+
      | Feature                  | API                                                                   | Description                                                                            |
      +==========================+=======================================================================+========================================================================================+
      | Asynchronous Infer       | ``ov::InferRequest::start_async``, ``ov::InferRequest::set_callback`` | Do asynchronous inference with callback.                                               |
      +--------------------------+-----------------------------------------------------------------------+----------------------------------------------------------------------------------------+
      | Model Operations         | ``ov::Output::get_shape``, ``ov::set_batch``                          | Manage the model, operate with its batch size. Set batch size using input image count. |
      +--------------------------+-----------------------------------------------------------------------+----------------------------------------------------------------------------------------+
      | Infer Request Operations | ``ov::InferRequest::get_input_tensor``                                | Get an input tensor.                                                                   |
      +--------------------------+-----------------------------------------------------------------------+----------------------------------------------------------------------------------------+
      | Tensor Operations        | ``ov::shape_size``, ``ov::Tensor::data``                              | Get a tensor shape size and its data.                                                  |
      +--------------------------+-----------------------------------------------------------------------+----------------------------------------------------------------------------------------+

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_inference_engine_samples_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/cpp/classification_sample_async/main.cpp 
         :language: cpp

How It Works
############

At startup, the sample application reads command line parameters and loads the specified model and input images (or a
folder with images) to the OpenVINO™ Runtime plugin. The batch size of the model is set according to the number of read images. The batch mode is an independent attribute on the asynchronous mode. Asynchronous mode works efficiently with any batch size.

Then, the sample creates an inference request object and assigns completion callback for it. In scope of the completion callback handling the inference request is executed again.

After that, the application starts inference for the first infer request and waits of 10th inference request execution being completed. The asynchronous mode might increase the throughput of the pictures.

When inference is done, the application outputs data to the standard output stream. You can place labels in .labels file near the model to get pretty output.

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

Run the application with the ``-h`` option to see the usage instructions:

.. code-block:: sh

   classification_sample_async -h

Usage instructions:

.. code-block:: sh

   [ INFO ] OpenVINO Runtime version ......... <version>
   [ INFO ] Build ........... <build>
   
   classification_sample_async [OPTION]
   Options:
   
       -h                      Print usage instructions.
       -m "<path>"             Required. Path to an .xml file with a trained model.
       -i "<path>"             Required. Path to a folder with images or path to image files: a .ubyte file for LeNet and a .bmp file for other models.
       -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify the HETERO plugin. Sample will look for a suitable plugin for the device specified.
   
   Available target devices: <devices>

To run the sample, you need to specify a model and image:

- You can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.
- You can use images from the media files collection available `here <https://storage.openvinotoolkit.org/data/test_data>`.

.. note::

   - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using ``mo`` with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.

   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.

   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

   - Stating flags that take only single option like `-m` multiple times, for example `./classification_sample_async -m model.xml -m model2.xml`, results in only the first value being used.

   - The sample supports NCHW model layout only.

Example
+++++++

1. Install the ``openvino-dev`` Python package to use Open Model Zoo Tools:

   .. code-block:: sh
      
      python -m pip install openvino-dev[caffe]
   

2. Download a pre-trained model using:

   .. code-block:: sh
      
      omz_downloader --name googlenet-v1
   

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

   .. code-block:: sh
      
      omz_converter --name googlenet-v1

4. Perform inference of ``dog.bmp`` using ``googlenet-v1`` model on a ``GPU``, for example:
   
   .. code-block:: sh
       
      classification_sample_async -m googlenet-v1.xml -i dog.bmp -d GPU

Sample Output
#############

.. code-block:: sh
   
   [ INFO ] OpenVINO Runtime version ......... <version>
   [ INFO ] Build ........... <build>
   [ INFO ]
   [ INFO ] Parsing input parameters
   [ INFO ] Files were added: 1
   [ INFO ]     /images/dog.bmp
   [ INFO ] Loading model files:
   [ INFO ] /models/googlenet-v1.xml
   [ INFO ] model name: GoogleNet
   [ INFO ]     inputs
   [ INFO ]         input name: data
   [ INFO ]         input type: f32
   [ INFO ]         input shape: {1, 3, 224, 224}
   [ INFO ]     outputs
   [ INFO ]         output name: prob
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {1, 1000}
   [ INFO ] Read input images
   [ INFO ] Set batch size 1
   [ INFO ] model name: GoogleNet
   [ INFO ]     inputs
   [ INFO ]         input name: data
   [ INFO ]         input type: u8
   [ INFO ]         input shape: {1, 224, 224, 3}
   [ INFO ]     outputs
   [ INFO ]         output name: prob
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {1, 1000}
   [ INFO ] Loading model to the device GPU
   [ INFO ] Create infer request
   [ INFO ] Start inference (asynchronous executions)
   [ INFO ] Completed 1 async request execution
   [ INFO ] Completed 2 async request execution
   [ INFO ] Completed 3 async request execution
   [ INFO ] Completed 4 async request execution
   [ INFO ] Completed 5 async request execution
   [ INFO ] Completed 6 async request execution
   [ INFO ] Completed 7 async request execution
   [ INFO ] Completed 8 async request execution
   [ INFO ] Completed 9 async request execution
   [ INFO ] Completed 10 async request execution
   [ INFO ] Completed async requests execution
   
   Top 10 results:
   
   Image /images/dog.bmp
   
   classid probability
   ------- -----------
   156     0.8935547
   218     0.0608215
   215     0.0217133
   219     0.0105667
   212     0.0018835
   217     0.0018730
   152     0.0018730
   157     0.0015745
   154     0.0012817
   220     0.0010099

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective

