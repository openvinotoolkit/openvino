# Model Creation C++ Sample {#openvino_inference_engine_samples_model_creation_sample_README}

@sphinxdirective

.. meta::
   :description: Learn how to create a model on the fly with a 
                 provided weights file and infer it later using Synchronous 
                 Inference Request (C++) API.


This sample demonstrates how to execute an synchronous inference using :doc:`model <openvino_docs_OV_UG_Model_Representation>` built on the fly which uses weights from LeNet classification model, which is known to work well on digit classification tasks.

You do not need an XML file to create a model. The API of ov::Model allows creating a model on the fly from the source code.

.. tab-set::

   .. tab-item:: Requirements 

      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+
      | Options                                                 | Values                                                                                          |
      +=========================================================+=================================================================================================+
      | Validated Models                                        | LeNet                                                                                           |
      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+
      | Model Format                                            | model weights file (\*.bin)                                                                     |
      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+
      | Validated images                                        | single-channel ``MNIST ubyte`` images                                                           |
      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+
      | Supported devices                                       | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                            |
      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+
      | Other language realization                              | :doc:`Python <openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README>` |
      +---------------------------------------------------------+-------------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API 

      The following C++ API is used in the application:

      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Feature                                  | API                                     | Description                           |
      +==========================================+=========================================+=======================================+
      | OpenVINO Runtime Info                    | ``ov::Core::get_versions``              | Get device plugins versions           |
      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Shape Operations                         | ``ov::Output::get_shape``,              | Operate with shape                    |
      |                                          | ``ov::Shape::size``,                    |                                       |
      |                                          | ``ov::shape_size``                      |                                       |
      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Tensor Operations                        | ``ov::Tensor::get_byte_size``,          | Get tensor byte size and its data     |
      |                                          | ``ov::Tensor:data``                     |                                       |
      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Model Operations                         | ``ov::set_batch``                       | Operate with model batch size         |
      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Infer Request Operations                 | ``ov::InferRequest::get_input_tensor``  | Get a input tensor                    |
      +------------------------------------------+-----------------------------------------+---------------------------------------+
      | Model creation objects                   | ``ov::opset8::Parameter``,              | Used to construct an OpenVINO model   |
      |                                          | ``ov::Node::output``,                   |                                       |
      |                                          | ``ov::opset8::Constant``,               |                                       |
      |                                          | ``ov::opset8::Convolution``,            |                                       |
      |                                          | ``ov::opset8::Add``,                    |                                       |
      |                                          | ``ov::opset1::MaxPool``,                |                                       |
      |                                          | ``ov::opset8::Reshape``,                |                                       |
      |                                          | ``ov::opset8::MatMul``,                 |                                       |
      |                                          | ``ov::opset8::Relu``,                   |                                       |
      |                                          | ``ov::opset8::Softmax``,                |                                       |
      |                                          | ``ov::descriptor::Tensor::set_names``,  |                                       |
      |                                          | ``ov::opset8::Result``,                 |                                       |
      |                                          | ``ov::Model``,                          |                                       |
      |                                          | ``ov::ParameterVector::vector``         |                                       |
      +------------------------------------------+-----------------------------------------+---------------------------------------+

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_inference_engine_samples_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/cpp/model_creation_sample/main.cpp 
         :language: cpp

How It Works
############

At startup, the sample application does the following:

- Reads command line parameters
- :doc:`Build a Model <openvino_docs_OV_UG_Model_Representation>` and passed weights file
- Loads the model and input data to the OpenVINO™ Runtime plugin
- Performs synchronous inference and processes output data, logging each step in a standard output stream

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

.. code-block:: console

   model_creation_sample <path_to_lenet_weights> <device>

.. note::

   - you can use LeNet model weights in the sample folder: ``lenet.bin`` with FP32 weights file
   - The ``lenet.bin`` with FP32 weights file was generated by :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` from the public LeNet model with the ``input_shape [64,1,28,28]`` parameter specified.
   
   The original model is available in the `Caffe* repository <https://github.com/BVLC/caffe/tree/master/examples/mnist>`__ on GitHub\*.


You can do inference of an image using a pre-trained model on a GPU using the following command:

.. code-block:: console
   
   model_creation_sample lenet.bin GPU

Sample Output
#############

The sample application logs each step in a standard output stream and outputs top-10 inference results.

.. code-block:: console
   
   [ INFO ] OpenVINO Runtime version ......... <version>
   [ INFO ] Build ........... <build>
   [ INFO ]
   [ INFO ] Device info:
   [ INFO ] GPU
   [ INFO ] Intel GPU plugin version ......... <version>
   [ INFO ] Build ........... <build>
   [ INFO ]
   [ INFO ]
   [ INFO ] Create model from weights: lenet.bin
   [ INFO ] model name: lenet
   [ INFO ]     inputs
   [ INFO ]         input name: NONE
   [ INFO ]         input type: f32
   [ INFO ]         input shape: {64, 1, 28, 28}
   [ INFO ]     outputs
   [ INFO ]         output name: output_tensor
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {64, 10}
   [ INFO ] Batch size is 10
   [ INFO ] model name: lenet
   [ INFO ]     inputs
   [ INFO ]         input name: NONE
   [ INFO ]         input type: u8
   [ INFO ]         input shape: {10, 28, 28, 1}
   [ INFO ]     outputs
   [ INFO ]         output name: output_tensor
   [ INFO ]         output type: f32
   [ INFO ]         output shape: {10, 10}
   [ INFO ] Compiling a model for the GPU device
   [ INFO ] Create infer request
   [ INFO ] Combine images in batch and set to input tensor
   [ INFO ] Start sync inference
   [ INFO ] Processing output tensor
   
   Top 1 results:
   
   Image 0
   
   classid probability label
   ------- ----------- -----
   0       1.0000000   0
   
   Image 1
   
   classid probability label
   ------- ----------- -----
   1       1.0000000   1
   
   Image 2
   
   classid probability label
   ------- ----------- -----
   2       1.0000000   2
   
   Image 3
   
   classid probability label
   ------- ----------- -----
   3       1.0000000   3
   
   Image 4
   
   classid probability label
   ------- ----------- -----
   4       1.0000000   4
   
   Image 5
   
   classid probability label
   ------- ----------- -----
   5       1.0000000   5
   
   Image 6
   
   classid probability label
   ------- ----------- -----
   6       1.0000000   6
   
   Image 7
   
   classid probability label
   ------- ----------- -----
   7       1.0000000   7
   
   Image 8
   
   classid probability label
   ------- ----------- -----
   8       1.0000000   8
   
   Image 9
   
   classid probability label
   ------- ----------- -----
   9       1.0000000   9
   


Deprecation Notice
##################

+--------------------+------------------+
| Deprecation Begins | June 1, 2020     |
+====================+==================+
| Removal Date       | December 1, 2020 |
+--------------------+------------------+

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective

