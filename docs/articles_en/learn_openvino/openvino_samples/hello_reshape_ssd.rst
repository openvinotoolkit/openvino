.. {#openvino_sample_hello_reshape_ssd}

Hello Reshape SSD Sample
========================


.. meta::
   :description: Learn how to do inference of object detection 
                 models using shape inference feature and Synchronous 
                 Inference Request API (Python, C++).


This sample demonstrates how to do synchronous inference of object detection models using :doc:`Shape Inference feature <openvino_docs_OV_UG_ShapeInference>`.  

Models with only one input and output are supported.


Requirements
####################

+-------------------+--------------------------------------------------------------------------------------+
| Options           | Values                                                                               |
+===================+======================================================================================+
| Validated Models  || :doc:`mobilenet-ssd <omz_models_model_mobilenet_ssd>`                               |
|                   || :doc:`person-detection-retail-0013 <omz_models_model_person_detection_retail_0013>` |
+-------------------+--------------------------------------------------------------------------------------+
| Validated Layout  | NCHW                                                                                 |
+-------------------+--------------------------------------------------------------------------------------+
| Model Format      | OpenVINO™ toolkit Intermediate Representation (.xml + .bin), ONNX (.onnx)            |
+-------------------+--------------------------------------------------------------------------------------+
| Supported devices | :doc:`All <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`                 |
+-------------------+--------------------------------------------------------------------------------------+


How It Works
####################

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous inference, and processes output data.  
As a result, the program creates an output image, logging each step in a standard output stream.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/python/hello_reshape_ssd/hello_reshape_ssd.py
                  :language: python

         .. tab-item:: API
      
            The following Python API is used in the application:
      
            +------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------+
            | Feature                            | API                                                                                                                                                                            | Description                          |
            +====================================+================================================================================================================================================================================+======================================+
            | Model Operations                   | `openvino.runtime.Model.reshape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape>`__ ,               | Managing of model                    |
            |                                    | `openvino.runtime.Model.input <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.input>`__ ,                   |                                      |
            |                                    | `openvino.runtime.Output.get_any_name <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Output.html#openvino.runtime.Output.get_any_name>`__ ,  |                                      |
            |                                    | `openvino.runtime.PartialShape <https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.PartialShape.html>`__                                          |                                      |
            +------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------+
      
            Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification Python* Sample <openvino_sample_hello_classification>`.
      

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::
      
         .. tab-item:: Sample Code

            .. scrollbox::

               .. doxygensnippet:: samples/cpp/hello_reshape_ssd/main.cpp 
                  :language: cpp

         .. tab-item:: API
      
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
      
            Basic OpenVINO™ Runtime API is covered by :doc:`Hello Classification C++ sample <openvino_sample_hello_classification>`.


You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Building
####################

To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>` section in OpenVINO™ Toolkit Samples guide.

Running
####################


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console
         
         python hello_reshape_ssd.py <path_to_model> <path_to_image> <device_name>

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console
         
         hello_reshape_ssd <path_to_model> <path_to_image> <device_name>


To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model 
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at 
  `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.

.. note::
  
   - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels 
     order. If you trained your model to work with RGB order, you need to manually 
     rearrange the default channels order in the sample or demo application or 
     reconvert your model using model conversion API with ``reverse_input_channels`` 
     argument specified. For more information about the argument, refer to 
     **When to Reverse Input Channels** section of 
     :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.
   - Before running the sample with a trained model, make sure the model is 
     converted to the intermediate representation (IR) format (\*.xml + \*.bin) 
     using :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

Example
++++++++++++++++++++

1. Download a pre-trained model:
2. If a model is not in the IR or ONNX format, it must be converted by using:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            import openvino as ov

            ov_model = ov.convert_model('./test_data/models/mobilenet-ssd')
            # or, when model is a Python model object
            ov_model = ov.convert_model(mobilenet-ssd)

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: console

            ovc ./test_data/models/mobilenet-ssd

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            mo --input_model <path_to_model>

4. Perform inference of an image, using a model on a ``GPU``, for example:

   .. tab-set::
   
      .. tab-item:: Python
         :sync: python
   
         .. code-block:: console
            
            python hello_reshape_ssd.py ./test_data/models/mobilenet-ssd.xml banana.jpg GPU
   
      .. tab-item:: C++
         :sync: cpp
   
         .. code-block:: console
            
            hello_reshape_ssd ./models/person-detection-retail-0013.xml person_detection.bmp GPU


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The sample application logs each step in a standard output stream and 
      creates an output image, drawing bounding boxes for inference results 
      with an over 50% confidence.
      
      .. code-block:: console
         
         [ INFO ] Creating OpenVINO Runtime Core
         [ INFO ] Reading the model: C:/test_data/models/mobilenet-ssd.xml
         [ INFO ] Reshaping the model to the height and width of the input image
         [ INFO ] Loading the model to the plugin
         [ INFO ] Starting inference in synchronous mode
         [ INFO ] Found: class_id = 52, confidence = 0.98, coords = (21, 98), (276, 210)
         [ INFO ] Image out.bmp was created!
         [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool


   .. tab-item:: C++
      :sync: cpp

      The application renders an image with detected objects enclosed in rectangles. 
      It outputs the list of classes of the detected objects along with the 
      respective confidence values and the coordinates of the rectangles to the 
      standard output stream.
      
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


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
