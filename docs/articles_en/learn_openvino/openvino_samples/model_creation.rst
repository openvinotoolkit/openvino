.. {#openvino_sample_model_creation}

Model Creation Sample
=====================


.. meta::
   :description: Learn how to create a model on the fly with a 
                 provided weights file and infer it later using Synchronous 
                 Inference Request API (Python, C++).


This sample demonstrates how to run inference using a :doc:`model <openvino_docs_OV_UG_Model_Representation>` 
built on the fly that uses weights from the LeNet classification model, which is 
known to work well on digit classification tasks. You do not need an XML file, 
the model is created from the source code on the fly.

Requirements
####################

+-------------------+----------------------------------------------------------------------+
| Options           | Values                                                               |
+===================+======================================================================+
| Validated Models  | LeNet                                                                |
+-------------------+----------------------------------------------------------------------+
| Model Format      | Model weights file (\*.bin)                                          |
+-------------------+----------------------------------------------------------------------+
| Supported devices | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>` |
+-------------------+----------------------------------------------------------------------+


How It Works
####################

At startup, the sample application reads command-line parameters, `builds a model <openvino_docs_OV_UG_Model_Representation>` 
and passes the weights file. Then, it loads the model and input data to the OpenVINO™ 
Runtime plugin. Finally, it performs synchronous inference and processes output 
data, logging each step in a standard output stream.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::
      
         .. tab-item:: Sample Code
      
            .. scrollbox::
      
               .. doxygensnippet:: samples/python/model_creation_sample/model_creation_sample.py  
                  :language: python

         .. tab-item:: API
      
            The following OpenVINO Python API is used in the application:
      
            +------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
            | Feature                                  | API                                                                                                                                                          | Description                                                                        |
            +==========================================+==============================================================================================================================================================+====================================================================================+
            | Model Operations                         | `openvino.runtime.Model <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__ ,                                    | Managing of model                                                                  |
            |                                          | `openvino.runtime.set_batch <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.set_batch.html>`__ ,                            |                                                                                    |
            |                                          | `openvino.runtime.Model.input <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input>`__   |                                                                                    |
            +------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
            | Opset operations                         | `openvino.runtime.op.Parameter <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.op.Parameter.html>`__ ,                      | Description of a model topology using OpenVINO Python API                          |
            |                                          | `openvino.runtime.op.Constant <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.op.Constant.html>`__ ,                        |                                                                                    |
            |                                          | `openvino.runtime.opset8.convolution <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.convolution.html>`__ ,          |                                                                                    |
            |                                          | `openvino.runtime.opset8.add <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.add.html>`__ ,                          |                                                                                    |
            |                                          | `openvino.runtime.opset1.max_pool <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset1.max_pool.html>`__ ,                |                                                                                    |
            |                                          | `openvino.runtime.opset8.reshape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.reshape.html>`__ ,                  |                                                                                    |
            |                                          | `openvino.runtime.opset8.matmul <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.matmul.html>`__ ,                    |                                                                                    |
            |                                          | `openvino.runtime.opset8.relu <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.relu.html>`__ ,                        |                                                                                    |
            |                                          | `openvino.runtime.opset8.softmax <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.opset8.softmax.html>`__                    |                                                                                    |
            +------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
      
            Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification Python* Sample <openvino_sample_hello_classification>`.

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::
      
         .. tab-item:: Sample Code

            .. important::

               **Deprecation Notice:** This sample has been deprecated since June 1, 2020. The date of removal was set to December 1, 2020.

            .. scrollbox::

               .. doxygensnippet:: samples/cpp/model_creation_sample/main.cpp 
                  :language: cpp

         .. tab-item:: API

            .. important::

               **Deprecation Notice:** This sample has been deprecated since June 1, 2020. The date of removal was set to December 1, 2020.

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
      
            Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_sample_hello_classification>`.


You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
####################

To build the sample, use instructions available at 
:doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section 
in OpenVINO™ Toolkit Samples guide.


Running
####################

To run the sample, you need to specify model weights and a device.


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python model_creation_sample.py <path_to_weights_file> <device_name>

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         model_creation_sample <path_to_weights_file> <device_name>


.. note::

   - This sample supports models with FP32 weights only.
   - The ``lenet.bin`` weights file is generated by 
     :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` 
     from the public LeNet model, with the ``input_shape [64,1,28,28]`` parameter specified.
   - The original model is available in the 
     `Caffe repository <https://github.com/BVLC/caffe/tree/master/examples/mnist>`__ on GitHub.

Example
++++++++++++++++++++

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python model_creation_sample.py lenet.bin GPU


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         model_creation_sample lenet.bin GPU


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The sample application logs each step in a standard output stream and outputs 10 inference results.

      .. code-block:: console

         [ INFO ] Creating OpenVINO Runtime Core
         [ INFO ] Loading the model using ngraph function with weights from lenet.bin
         [ INFO ] Loading the model to the plugin
         [ INFO ] Starting inference in synchronous mode
         [ INFO ] Top 1 results: 
         [ INFO ] Image 0
         [ INFO ]        
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 0       1.0000000   0
         [ INFO ]
         [ INFO ] Image 1
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 1       1.0000000   1
         [ INFO ]
         [ INFO ] Image 2
         [ INFO ] 
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 2       1.0000000   2
         [ INFO ]
         [ INFO ] Image 3
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 3       1.0000000   3
         [ INFO ]
         [ INFO ] Image 4
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 4       1.0000000   4
         [ INFO ]
         [ INFO ] Image 5
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 5       1.0000000   5
         [ INFO ]
         [ INFO ] Image 6
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 6       1.0000000   6
         [ INFO ]
         [ INFO ] Image 7
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 7       1.0000000   7
         [ INFO ]
         [ INFO ] Image 8
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 8       1.0000000   8
         [ INFO ]
         [ INFO ] Image 9
         [ INFO ]
         [ INFO ] classid probability label
         [ INFO ] -------------------------
         [ INFO ] 9       1.0000000   9
         [ INFO ]
         [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
      
   .. tab-item:: C++
      :sync: cpp

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


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
