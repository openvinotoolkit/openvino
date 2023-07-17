# Preprocessing {#openvino_2_0_preprocessing}

@sphinxdirective

.. meta::
   :description: In OpenVINO™ API 2.0 each preprocessing or post-processing 
                 operation is integrated directly into the model and compiled 
                 together with the inference graph.


This guide introduces how preprocessing works in API 2.0 by a comparison with preprocessing in the previous Inference Engine API. It also demonstrates how to migrate preprocessing scenarios from Inference Engine to API 2.0 via code samples.

How Preprocessing Works in API 2.0
##################################

Inference Engine API contains preprocessing capabilities in the `InferenceEngine::CNNNetwork <classInferenceEngine_1_1CNNNetwork.html#doxid-class-inference-engine-1-1-c-n-n-network>`__ class. Such preprocessing information is not a part of the main inference graph executed by :doc:`OpenVINO devices <openvino_docs_OV_UG_Working_with_devices>`. Therefore, it is stored and executed separately before the inference stage:

* Preprocessing operations are executed on the CPU for most OpenVINO inference plugins. Thus, instead of occupying accelerators, they keep the CPU busy with computational tasks.
* Preprocessing information stored in `InferenceEngine::CNNNetwork <classInferenceEngine_1_1CNNNetwork.html#doxid-class-inference-engine-1-1-c-n-n-network>`__ is lost when saving back to the OpenVINO IR file format.

API 2.0 introduces a :doc:`new way of adding preprocessing operations to the model <openvino_docs_OV_UG_Preprocessing_Overview>` - each preprocessing or post-processing operation is integrated directly into the model and compiled together with the inference graph:

* API 2.0 first adds preprocessing operations by using `ov::preprocess::PrePostProcessor <classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor>`__,
* and then compiles the model on the target by using `ov::Core::compile_model <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__.

Having preprocessing operations as a part of an OpenVINO opset makes it possible to read and serialize a preprocessed model as the OpenVINO™ IR file format.

More importantly, API 2.0 does not assume any default layouts as Inference Engine did. For example, both ``{ 1, 224, 224, 3 }`` and ``{ 1, 3, 224, 224 }`` shapes are supposed to be in the `NCHW` layout, while only the latter is. Therefore, some preprocessing capabilities in the API require layouts to be set explicitly. To learn how to do it, refer to the :doc:`Layout overview <openvino_docs_OV_UG_Layout_Overview>`. For example, to perform image scaling by partial dimensions ``H`` and ``W``, preprocessing needs to know what dimensions ``H`` and ``W`` are.

.. note::

   Use model conversion API preprocessing capabilities to insert preprocessing operations in your model for optimization. Thus, the application does not need to read the model and set preprocessing repeatedly. You can use the :doc:`model caching feature <openvino_docs_OV_UG_Model_caching_overview>` to improve the time-to-inference.

The following sections demonstrate how to migrate preprocessing scenarios from Inference Engine API to API 2.0.
The snippets assume that you need to preprocess a model input with the ``tensor_name`` in Inference Engine API, using ``operation_name`` to address the data.

Preparation: Import Preprocessing in Python
###########################################

In order to utilize preprocessing, the following imports must be added.

**Inference Engine API**


.. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
    :language: python
    :fragment: imports


**API 2.0**


.. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
    :language: python
    :fragment: ov_imports


| There are two different namespaces:
| * ``runtime``, which contains API 2.0 classes;
| * and ``preprocess``, which provides Preprocessing API.

Using Mean and Scale Values
###########################

**Inference Engine API**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: mean_scale

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: mean_scale

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: c_api_ppp


**API 2.0**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: ov_mean_scale

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: ov_mean_scale

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: ov_mean_scale


Converting Precision and Layout
###############################

**Inference Engine API**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: conversions

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: conversions

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: c_api_ppp


**API 2.0**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: ov_conversions

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: ov_conversions

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: ov_conversions


Using Image Scaling
####################

**Inference Engine API**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: image_scale

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: image_scale

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: c_api_ppp


**API 2.0**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: ov_image_scale

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: ov_image_scale

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: ov_image_scale


Converting Color Space
++++++++++++++++++++++

**API 2.0**


.. tab-set::

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
           :language: python
           :fragment: ov_color_space

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
            :language: cpp
            :fragment: ov_color_space

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.c
           :language: c
           :fragment: ov_color_space


Additional Resources
####################

- :doc:`Preprocessing details <openvino_docs_OV_UG_Preprocessing_Details>`
- :doc:`NV12 classification sample <openvino_inference_engine_samples_hello_nv12_input_classification_README>`

@endsphinxdirective
