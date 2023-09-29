# OpenVINO Extensibility Mechanism {#openvino_docs_Extensibility_UG_Intro}

@sphinxdirective

.. meta::
   :description: Explore OpenVINO™ Extensibility API, which allows adding 
                 support for models with custom operations and their further implementation 
                 in applications.

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops
   openvino_docs_Extensibility_UG_Frontend_Extensions
   openvino_docs_Extensibility_UG_GPU
   
.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_transformations
   OpenVINO Plugin Developer Guide <openvino_docs_ie_plugin_dg_overview>


The Intel® Distribution of OpenVINO™ toolkit supports neural-network models trained with various frameworks, including
TensorFlow, PyTorch, ONNX, TensorFlow Lite, and PaddlePaddle (OpenVINO support for Apache MXNet, Caffe, and Kaldi is currently 
being deprecated and will be removed entirely in the future). The list of supported operations is different for each of the supported frameworks. 
To see the operations supported by your framework, refer to :doc:`Supported Framework Operations <openvino_resources_supported_operations_frontend>`.

Custom operations, which are not included in the list, are not recognized by OpenVINO out-of-the-box. The need for custom operation may appear in two cases:

1. A new or rarely used regular framework operation is not supported in OpenVINO yet.
2. A new user operation that was created for some specific model topology by the author of the model using framework extension capabilities.

Importing models with such operations requires additional steps. This guide illustrates the workflow for running inference on models featuring custom operations. This allows plugging in your own implementation for them. OpenVINO Extensibility API enables adding support for those custom operations and using one implementation for Model Optimizer and OpenVINO Runtime.

Defining a new custom operation basically consists of two parts:

1. Definition of operation semantics in OpenVINO, the code that describes how this operation should be inferred consuming input tensor(s) and producing output tensor(s). The implementation of execution kernels for :doc:`GPU <openvino_docs_Extensibility_UG_GPU>` is described in separate guides.

2. Mapping rule that facilitates conversion of framework operation representation to OpenVINO defined operation semantics.

The first part is required for inference. The second part is required for successful import of a model containing such operations from the original framework model format. There are several options to implement each part. The following sections will describe them in detail.

Definition of Operation Semantics
#################################

If the custom operation can be mathematically represented as a combination of exiting OpenVINO operations and such decomposition gives desired performance, then low-level operation implementation is not required. Refer to the latest OpenVINO operation set, when deciding feasibility of such decomposition. You can use any valid combination of exiting operations. The next section of this document describes the way to map a custom operation.

If such decomposition is not possible or appears too bulky with a large number of constituent operations that do not perform well, then a new class for the custom operation should be implemented, as described in the :doc:`Custom Operation Guide <openvino_docs_Extensibility_UG_add_openvino_ops>`.

You might prefer implementing a custom operation class if you already have a generic C++ implementation of operation kernel. Otherwise, try to decompose the operation first, as described above. Then, after verifying correctness of inference and resulting performance, you may move on to optional implementation of Bare Metal C++.

Mapping from Framework Operation
################################

Mapping of custom operation is implemented differently, depending on model format used for import. You may choose one of the following:

1. If a model is represented in the ONNX (including models exported from PyTorch in ONNX), TensorFlow Lite, PaddlePaddle or TensorFlow formats, then one of the classes from :doc:`Frontend Extension API <openvino_docs_Extensibility_UG_Frontend_Extensions>` should be used. It consists of several classes available in C++ which can be used with the ``--extensions`` option in Model Optimizer or when a model is imported directly to OpenVINO runtime using the ``read_model`` method. Python API is also available for runtime model import.

2. If a model is represented in the Caffe, Kaldi or MXNet formats (as legacy frontends), then :doc:`[Legacy] Model Optimizer Extensions <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer>` should be used. This approach is available for model conversion in Model Optimizer only.

Existing of two approaches simultaneously is explained by two different types of frontends used for model conversion in OpenVINO: new frontends (ONNX, PaddlePaddle, TensorFlow Lite, and TensorFlow) and legacy frontends (Caffe, Kaldi, and Apache MXNet). Model Optimizer can use both frontends in contrast to the direct import of model with ``read_model`` method which can use new frontends only. Follow one of the appropriate guides referenced above to implement mappings depending on framework frontend.

If you are implementing extensions for new ONNX, PaddlePaddle, TensorFlow Lite or TensorFlow frontends and plan to use the ``--extensions`` option in Model Optimizer for model conversion, then the extensions should be:

1. Implemented in C++ only.

2. Compiled as a separate shared library (see details on how to do this further in this guide).

Model Optimizer does not support new frontend extensions written in Python API.

Remaining part of this guide describes application of Frontend Extension API for new frontends.

Registering Extensions
######################

A custom operation class and a new mapping frontend extension class object should be registered to be usable in OpenVINO runtime.

.. note:: 
   This documentation is derived from the `Template extension <https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension/new>`__, which demonstrates the details of extension development. It is based on minimalistic ``Identity`` operation that is a placeholder for your real custom operation. Review the complete, fully compilable code to see how it works.

Use the ``:ref:`ov::Core::add_extension <doxid-classov_1_1_core_1a68d0dea1cbcd42a67bea32780e32acea>``` method to load the extensions to the ``:ref:`ov::Core <doxid-classov_1_1_core>``` object. This method allows loading library with extensions or extensions from the code.

Load Extensions to Core
+++++++++++++++++++++++

Extensions can be loaded from a code with the  ``:ref:`ov::Core::add_extension <doxid-classov_1_1_core_1a68d0dea1cbcd42a67bea32780e32acea>``` method:

.. tab-set::

   .. tab-item:: Python
      :sync: py
 
      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [add_extension]

   .. tab-item:: C++
      :sync: cpp
 
      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [add_extension]
   

The ``Identity`` is a custom operation class defined in :doc:`Custom Operation Guide <openvino_docs_Extensibility_UG_add_openvino_ops>`. This is sufficient to enable reading OpenVINO IR which uses the ``Identity`` extension operation emitted by Model Optimizer. In order to load original model directly to the runtime, add a mapping extension:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [add_frontend_extension]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [add_frontend_extension]

When Python API is used, there is no way to implement a custom OpenVINO operation. Even if custom OpenVINO operation is implemented in C++ and loaded into the runtime by a shared library, there is still no way to add a frontend mapping extension that refers to this custom operation. In this case, use C++ shared library approach to implement both operations semantics and framework mapping.

Python can still be used to map and decompose operations when only operations from the standard OpenVINO operation set are used.

.. _create_a_library_with_extensions:
 
Create a Library with Extensions
++++++++++++++++++++++++++++++++

An extension library should be created in the following cases:

* Conversion of a model with custom operations in Model Optimizer.
* Loading a model with custom operations in a Python application. This applies to both framework model and OpenVINO IR.
* Loading models with custom operations in tools that support loading extensions from a library, for example the ``benchmark_app``.

To create an extension library, for example, to load the extensions into Model Optimizer, perform the following:

1. Create an entry point for extension library. OpenVINO provides the ``:ref:`OPENVINO_CREATE_EXTENSIONS() <doxid-core_2include_2openvino_2core_2extension_8hpp_1acdadcfa0eff763d8b4dadb8a9cb6f6e6>``` macro, which allows to define an entry point to a library with OpenVINO Extensions.
This macro should have a vector of all OpenVINO Extensions as an argument.

Based on that, the declaration of an extension class might look like the following:

.. doxygensnippet:: ./src/core/template_extension/new/ov_extension.cpp
   :language: cpp
   :fragment: [ov_extension:entry_point]

2. Configure the build of your extension library, using the following CMake script:

.. doxygensnippet:: ./src/core/template_extension/new/CMakeLists.txt
   :language: cpp
   :fragment: [cmake:extension]

This CMake script finds OpenVINO, using the ``find_package`` CMake command.

3. Build the extension library, running the commands below:

.. code-block:: sh
   
   $ cd src/core/template_extension/new
   $ mkdir build
   $ cd build
   $ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
   $ cmake --build .


4. After the build, you may use the path to your extension library to load your extensions to OpenVINO Runtime:

.. tab-set::
   
   .. tab-item:: Python
      :sync: py
 
      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [add_extension_lib]

   .. tab-item:: C++
      :sync: cpp 

      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [add_extension_lib]


See Also
########

* :doc:`OpenVINO Transformations <openvino_docs_transformations>`
* :doc:`Using OpenVINO Runtime Samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Hello Shape Infer SSD sample <openvino_inference_engine_samples_hello_reshape_ssd_README>`

@endsphinxdirective
