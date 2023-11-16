# Inference Pipeline {#openvino_2_0_inference_pipeline}

@sphinxdirective

.. meta::
   :description: The inference pipeline is a set of steps to be 
                 performed in a specific order to infer models with OpenVINO™ 
                 Runtime.


To infer models with OpenVINO™ Runtime, you usually need to perform the following steps in the application pipeline:

1. `Create a Core object <#create-a-core-object>`__.

   * 1.1. `(Optional) Load extensions <#optional-load-extensions>`__

2. `Read a model from a drive <#read-a-model-from-a-drive>`__.

   * 2.1. `(Optional) Perform model preprocessing <#optional-perform-model-preprocessing>`__.

3. `Load the model to the device <#load-the-model-to-the-device>`__.
4. `Create an inference request <#create-an-inference-request>`__.
5. `Fill input tensors with data <#fill-input-tensors-with-data>`__.
6. `Start inference <#start-inference>`__.
7. `Process the inference results <#process-the-inference-results>`__.

Based on the steps, the following code demonstrates how to change the application code to migrate to API 2.0.

1. Create a Core Object
#######################

**Inference Engine API**

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ie_common.cpp
         :language: cpp
         :fragment: ie:create_core

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ie_common.c
         :language: cpp
         :fragment: ie:create_core

**API 2.0**

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_common.py
         :language: python
         :fragment: ov_api_2_0:create_core

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_common.cpp
         :language: cpp
         :fragment: ov_api_2_0:create_core

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ov_common.c
         :language: cpp
         :fragment: ov_api_2_0:create_core


1.1 (Optional) Load Extensions
++++++++++++++++++++++++++++++

To load a model with custom operations, you need to add extensions for these operations. 
It is highly recommended to use :doc:`OpenVINO Extensibility API <openvino_docs_Extensibility_UG_Intro>` 
to write extensions. However, you can also load the old extensions to the new OpenVINO™ Runtime:

**Inference Engine API**

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ie_common.cpp
         :language: cpp
         :fragment: ie:load_old_extension

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ie_common.c
         :language: cpp
         :fragment: ie:load_old_extension


**API 2.0**

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_common.py
         :language: python
         :fragment: ov_api_2_0:load_old_extension

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_common.cpp
         :language: cpp
         :fragment: ov_api_2_0:load_old_extension

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ov_common.c
         :language: cpp
         :fragment: ov_api_2_0:load_old_extension


2. Read a Model from a Drive
############################

**Inference Engine API**

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ie_common.cpp
         :language: cpp
         :fragment: ie:read_model

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ie_common.c
         :language: cpp
         :fragment: ie:read_model


**API 2.0**

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_common.py
         :language: python
         :fragment: ov_api_2_0:read_model

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_common.cpp
         :language: cpp
         :fragment: ov_api_2_0:read_model

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ov_common.c
         :language: cpp
         :fragment: ov_api_2_0:read_model


Reading a model has the same structure as the example in the :doc:`model creation migration guide <openvino_2_0_model_creation>`.

You can combine reading and compiling a model into a single call ``ov::Core::compile_model(filename, devicename)``.


2.1 (Optional) Perform Model Preprocessing
++++++++++++++++++++++++++++++++++++++++++

When the application input data does not perfectly match the model input format, 
preprocessing may be necessary. See :doc:`preprocessing in API 2.0 <openvino_2_0_preprocessing>` for more details.


3. Load the Model to the Device
###############################

**Inference Engine API**

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ie_common.cpp
         :language: cpp
         :fragment: ie:compile_model

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ie_common.c
         :language: cpp
         :fragment: ie:compile_model


**API 2.0**

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_common.py
         :language: python
         :fragment: ov_api_2_0:compile_model

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_common.cpp
         :language: cpp
         :fragment: ov_api_2_0:compile_model

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ov_common.c
         :language: cpp
         :fragment: ov_api_2_0:compile_model


If you need to configure devices with additional parameters for OpenVINO Runtime, refer to :doc:`Configuring Devices <openvino_2_0_configure_devices>`.


4. Create an Inference Request
##############################

**Inference Engine API**

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ie_common.cpp
         :language: cpp
         :fragment: ie:create_infer_request

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ie_common.c
         :language: cpp
         :fragment: ie:create_infer_request


**API 2.0**

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_common.py
         :language: python
         :fragment: ov_api_2_0:create_infer_request

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_common.cpp
         :language: cpp
         :fragment: ov_api_2_0:create_infer_request

   .. tab-item:: C
      :sync: c

      .. doxygensnippet:: docs/snippets/ov_common.c
         :language: cpp
         :fragment: ov_api_2_0:create_infer_request


5. Fill Input Tensors with Data
###############################

**Inference Engine API**

The Inference Engine API fills inputs with data of the ``I32`` precision (**not** aligned with the original model):

.. tab-set::

   .. tab-item:: IR v10
      :sync: ir-v10

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_input_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_input_tensor

   .. tab-item:: IR v11
      :sync: ir-v11

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_input_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_input_tensor

   .. tab-item:: ONNX
      :sync: onnx

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_input_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_input_tensor


   .. tab-item:: Model created in code
      :sync: model

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_input_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_input_tensor


**API 2.0**

API 2.0 fills inputs with data of the ``I64`` precision (aligned with the original model):

.. tab-set::

   .. tab-item:: IR v10
      :sync: ir-v10

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_input_tensor_v10

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_v10

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_v10

   .. tab-item:: IR v11
      :sync: ir-v11

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned

   .. tab-item:: ONNX
      :sync: onnx

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned


   .. tab-item:: Model created in code
      :sync: model-created-in-code

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_input_tensor_aligned


6. Start Inference
##################

**Inference Engine API**

.. tab-set::

   .. tab-item:: Sync
      :sync: sync

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:inference

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:inference

   .. tab-item:: Async
      :sync: async

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:start_async_and_wait

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:start_async_and_wait


**API 2.0**

.. tab-set::

   .. tab-item:: Sync
      :sync: sync

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:inference

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:inference

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:inference

   .. tab-item:: Async
      :sync: async

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:start_async_and_wait

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:start_async_and_wait

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:start_async_and_wait


7. Process the Inference Results
################################

**Inference Engine API**

The Inference Engine API processes outputs as they are of the ``I32`` precision (**not** aligned with the original model):

.. tab-set::

   .. tab-item:: IR v10
      :sync: ir-v10

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_output_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_output_tensor

   .. tab-item:: IR v11
      :sync: ir-v11

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_output_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_output_tensor

   .. tab-item:: ONNX
      :sync: onnx

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_output_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_output_tensor


   .. tab-item:: Model created in code
      :sync: model

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ie_common.cpp
               :language: cpp
               :fragment: ie:get_output_tensor

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ie_common.c
               :language: cpp
               :fragment: ie:get_output_tensor


**API 2.0**

API 2.0 processes outputs as they are of:

* the ``I32`` precision (**not** aligned with the original model) for OpenVINO IR v10 models, to match the :ref:`old behavior <differences_api20_ie>`.
* the ``I64`` precision (aligned with the original model) for OpenVINO IR v11, ONNX, ov::Model, PaddlePaddle and TensorFlow models, to match the :ref:`new behavior <differences_api20_ie>`.

.. tab-set::

   .. tab-item:: IR v10
      :sync: ir-v10

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_output_tensor_v10

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_v10

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_v10

   .. tab-item:: IR v11
      :sync: ir-v11

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned

   .. tab-item:: ONNX
      :sync: onnx

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned


   .. tab-item:: Model created in code
      :sync: model-created-in-code

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/ov_common.py
               :language: python
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/ov_common.cpp
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned

         .. tab-item:: C
            :sync: c

            .. doxygensnippet:: docs/snippets/ov_common.c
               :language: cpp
               :fragment: ov_api_2_0:get_output_tensor_aligned


@endsphinxdirective
