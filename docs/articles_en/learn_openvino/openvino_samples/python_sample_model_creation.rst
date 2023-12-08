.. {#openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README}

Model Creation Python Sample
============================


.. meta::
   :description: Learn how to create a model on the fly with a 
                 provided weights file and infer it later using Synchronous 
                 Inference Request (Python) API.


This sample demonstrates how to run inference using a :doc:`model <openvino_docs_OV_UG_Model_Representation>` built on the fly that uses weights from the LeNet classification model, which is known to work well on digit classification tasks. You do not need an XML file, the model is created from the source code on the fly.

.. tab-set::

   .. tab-item:: Requirements 

      +------------------------------------------------+-----------------------------------------------------------------------------+
      | Options                                        | Values                                                                      |
      +================================================+=============================================================================+
      | Validated Models                               | LeNet                                                                       |
      +------------------------------------------------+-----------------------------------------------------------------------------+
      | Model Format                                   | Model weights file (\*.bin)                                                 |
      +------------------------------------------------+-----------------------------------------------------------------------------+
      | Supported devices                              | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`        |
      +------------------------------------------------+-----------------------------------------------------------------------------+
      | Other language realization                     | :doc:`C++ <openvino_inference_engine_samples_model_creation_sample_README>` |
      +------------------------------------------------+-----------------------------------------------------------------------------+

   .. tab-item:: Python API 

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

      Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification Python* Sample <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>`.

   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/python/model_creation_sample/model_creation_sample.py  
         :language: python

How It Works
############

At startup, the sample application does the following:

- Reads command line parameters
- :doc:`Build a Model <openvino_docs_OV_UG_Model_Representation>` and passed weights file
- Loads the model and input data to the OpenVINO™ Runtime plugin
- Performs synchronous inference and processes output data, logging each step in a standard output stream

You can see the explicit description of each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
#######

To run the sample, you need to specify model weights and device.

.. code-block:: console
   
   python model_creation_sample.py <path_to_model> <device_name>

.. note::
   
   - This sample supports models with FP32 weights only.
   
   - The ``lenet.bin`` weights file was generated by :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` from the public LeNet model with the ``input_shape [64,1,28,28]`` parameter specified.  
   
   - The original model is available in the `Caffe* repository <https://github.com/BVLC/caffe/tree/master/examples/mnist>`__ on GitHub\*.

For example:

.. code-block:: console
   
   python model_creation_sample.py lenet.bin GPU

Sample Output
#############

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

See Also
########

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Model Downloader <omz_tools_downloader>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`


