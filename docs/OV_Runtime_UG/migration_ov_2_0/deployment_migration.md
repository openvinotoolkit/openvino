# Installation & Deployment {#openvino_2_0_deployment}

@sphinxdirective

.. meta::
   :description: OpenVINO™ API 2.0 focuses on the use of development tools and 
                 deployment of applications, it also simplifies migration from 
                 different frameworks to OpenVINO.


One of the main concepts for OpenVINO™ API 2.0 is being "easy to use", which includes:

* Simplification of migration from different frameworks to OpenVINO.
* Organization of OpenVINO.
* Usage of development tools.
* Development and deployment of OpenVINO-based applications.


To accomplish that, the 2022.1 release OpenVINO introduced significant changes to the installation 
and deployment processes. Further changes were implemented in 2023.1, aiming at making the installation
process even simpler.

.. tip::

   These instructions are largely deprecated and should be used for versions prior to 2023.1.

   The OpenVINO Development Tools package is being deprecated and will be discontinued entirely in 2025.
   With this change, the OpenVINO Runtime package has become the default choice for installing the 
   software. It now includes all components necessary to utilize OpenVINO's functionality. 



The Installer Package Contains OpenVINO™ Runtime Only
#####################################################

Since OpenVINO 2022.1, development tools have been distributed only via `PyPI <https://pypi.org/project/openvino-dev/>`__, and are no longer included in the OpenVINO installer package. For a list of these components, refer to the :doc:`installation overview <openvino_docs_install_guides_overview>` guide. Benefits of this approach include:

* simplification of the user experience - in previous versions, installation and usage of OpenVINO Development Tools differed from one distribution type to another (the OpenVINO installer vs. PyPI),
* ensuring that dependencies are handled properly via the PIP package manager, and support virtual environments of development tools.

The structure of the OpenVINO 2022.1 installer package has been organized as follows:

* The ``runtime`` folder includes headers, libraries and CMake interfaces.
* The ``tools`` folder contains :doc:`the compile tool <openvino_ecosystem>`, :doc:`deployment manager <openvino_docs_install_guides_deployment_manager_tool>`, and a set of ``requirements.txt`` files with links to the corresponding versions of the ``openvino-dev`` package.
* The ``python`` folder contains the Python version for OpenVINO Runtime.

Installing OpenVINO Development Tools via PyPI
##############################################

Since OpenVINO Development Tools is no longer in the installer package, the installation process has also changed. This section describes it through a comparison with previous versions.

For Versions Prior to 2022.1
++++++++++++++++++++++++++++

In previous versions, OpenVINO Development Tools was a part of the main package. After the package was installed, to convert models (for example, TensorFlow), you needed to install additional dependencies by using the requirement files, such as ``requirements_tf.txt``, install Post-Training Optimization tool and Accuracy Checker tool via the ``setup.py`` scripts, and then use the ``setupvars`` scripts to make the tools available to the following command:

.. code-block:: sh

   $ mo.py -h


For 2022.1 and After (prior to 2023.1)
++++++++++++++++++++++++++++++++++++++++++

In OpenVINO 2022.1 and later, you can install the development tools only from a `PyPI <https://pypi.org/project/openvino-dev/>`__ repository, using the following command (taking TensorFlow as an example):

.. code-block:: sh

   $ python3 -m pip install -r <INSTALL_DIR>/tools/requirements_tf.txt 


This will install all the development tools and additional components necessary to work with TensorFlow via the ``openvino-dev`` package (see **Step 4. Install the Package** on the `PyPI page <https://pypi.org/project/openvino-dev/>`__ for parameters of other frameworks).

Then, the tools can be used by commands like:

.. code-block:: sh

   $ mo -h
   $ pot -h


Installation of any other dependencies is not required. For more details on the installation steps, see the 
`Install OpenVINO Development Tools <https://docs.openvino.ai/2023.0/openvino_docs_install_guides_install_dev_tools.html>`__ prior to OpenVINO 2023.1.

Interface Changes for Building C/C++ Applications
#################################################

The new OpenVINO Runtime with its API 2.0 has also brought some changes for building C/C++ applications.

CMake Interface
++++++++++++++++++++

The CMake interface has been changed as follows:

**With Inference Engine of previous versions**:

.. code-block:: cmake

   find_package(InferenceEngine REQUIRED)
   find_package(ngraph REQUIRED)
   add_executable(ie_ngraph_app main.cpp)
   target_link_libraries(ie_ngraph_app PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})


**With OpenVINO Runtime 2022.1 (API 2.0)**:

.. code-block:: cmake

   find_package(OpenVINO REQUIRED)
   add_executable(ov_app main.cpp)
   target_link_libraries(ov_app PRIVATE openvino::runtime)

   add_executable(ov_c_app main.c)
   target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)


Native Interfaces
++++++++++++++++++++

It is possible to build applications without the CMake interface by using: MSVC IDE, UNIX makefiles, and any other interface, which has been changed as shown here:

**With Inference Engine of previous versions**:

.. tab-set::

   .. tab-item:: Include dirs
      :sync: include-dirs
   
      .. code-block:: sh
   
         <INSTALL_DIR>/deployment_tools/inference_engine/include
         <INSTALL_DIR>/deployment_tools/ngraph/include
   
   .. tab-item:: Path to libs
      :sync: path-libs
   
      .. code-block:: sh
   
         <INSTALL_DIR>/deployment_tools/inference_engine/lib/intel64/Release
         <INSTALL_DIR>/deployment_tools/ngraph/lib/
   
   .. tab-item:: Shared libs
      :sync: shared-libs
   
      .. code-block:: sh
   
         // UNIX systems
         inference_engine.so ngraph.so
   
         // Windows
         inference_engine.dll ngraph.dll
   
   .. tab-item:: (Windows) .lib files
      :sync: windows-lib-files
   
      .. code-block:: sh
   
         ngraph.lib
         inference_engine.lib

**With OpenVINO Runtime 2022.1 (API 2.0)**:

.. tab-set::

   .. tab-item:: Include dirs
      :sync: include-dirs
   
      .. code-block:: sh
   
         <INSTALL_DIR>/runtime/include
   
   .. tab-item:: Path to libs
      :sync: path-libs
   
      .. code-block:: sh
   
         <INSTALL_DIR>/runtime/lib/intel64/Release
   
   .. tab-item:: Shared libs
      :sync: shared-libs
   
      .. code-block:: sh
   
         // UNIX systems
         openvino.so
   
         // Windows
         openvino.dll
   
   .. tab-item:: (Windows) .lib files
      :sync: windows-lib-files
   
      .. code-block:: sh
   
         openvino.lib
   

Clearer Library Structure for Deployment
########################################

OpenVINO 2022.1 introduced a reorganization of the libraries, to make deployment easier. In the previous versions, it was required to use several libraries to perform deployment steps. Now you can just use ``openvino`` or ``openvino_c`` based on your developing language,  with the necessary plugins to complete your task. For example, ``openvino_intel_cpu_plugin`` and ``openvino_ir_frontend`` plugins will enable loading OpenVINO IRs and performing inference on the CPU device (for more details, see the :doc:`Local distribution with OpenVINO <openvino_docs_deploy_local_distribution>`).

Below are detailed comparisons of the library structure between OpenVINO 2022.1 and the previous versions:

* Starting with 2022.1 release, a single core library with all the functionalities (``openvino`` for C++ Runtime, ``openvino_c`` for Inference Engine API C interface) is used, instead of the previous core libraries which contained ``inference_engine``, ``ngraph``, ``inference_engine_transformations`` and ``inference_engine_lp_transformations``.
* The optional ``inference_engine_preproc`` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat <classInferenceEngine_1_1PreProcessInfo.html#doxid-class-inference-engine-1-1-pre-process-info-1a3a10ba0d562a2268fe584d4d2db94cac>`__ or `InferenceEngine::PreProcessInfo::setResizeAlgorithm <classInferenceEngine_1_1PreProcessInfo.html#doxid-class-inference-engine-1-1-pre-process-info-1a0c083c43d01c53c327f09095e3e3f004>`__ is used) has been renamed to ``openvino_gapi_preproc`` and deprecated in 2022.1. For more details, see the :doc:`Preprocessing capabilities of OpenVINO API 2.0 <openvino_2_0_preprocessing>`.

* The libraries of plugins have been renamed as follows:

  * ``openvino_intel_cpu_plugin`` is used for :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>` device instead of ``MKLDNNPlugin``.
  * ``openvino_intel_gpu_plugin`` is used for :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>` device instead of ``clDNNPlugin``.
  * ``openvino_auto_plugin`` is used for :doc:`Auto-Device Plugin <openvino_docs_OV_UG_supported_plugins_AUTO>`.

* The plugins for reading and converting models have been changed as follows:

  * ``openvino_ir_frontend`` is used to read IRs instead of ``inference_engine_ir_reader``.
  * ``openvino_onnx_frontend`` is used to read ONNX models instead of ``inference_engine_onnx_reader`` (with its dependencies).
  * ``openvino_paddle_frontend`` is added in 2022.1 to read PaddlePaddle models.

<!-----
Older versions of OpenVINO had several core libraries and plugin modules:
- Core: ``inference_engine``, ``ngraph``, ``inference_engine_transformations``, ``inference_engine_lp_transformations``
- Optional ``inference_engine_preproc`` preprocessing library (if ``InferenceEngine::PreProcessInfo::setColorFormat`` or ``InferenceEngine::PreProcessInfo::setResizeAlgorithm`` are used)
- Plugin libraries:
 - ``MKLDNNPlugin`` for :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>` device
 - ``clDNNPlugin`` for :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>` device
 - ``MultiDevicePlugin`` for :doc:`Multi-device execution <openvino_docs_OV_UG_Running_on_multiple_devices>`
 - others
- Plugins to read and convert a model:
 - ``inference_engine_ir_reader`` to read OpenVINO IR
 - ``inference_engine_onnx_reader`` (with its dependencies) to read ONNX models
Now, the modularity is more clear:
- A single core library with all the functionality ``openvino`` for C++ runtime
- ``openvino_c`` with Inference Engine API C interface
- **Deprecated** Optional ``openvino_gapi_preproc`` preprocessing library (if ``InferenceEngine::PreProcessInfo::setColorFormat`` or ``InferenceEngine::PreProcessInfo::setResizeAlgorithm`` are used)
 - Use :doc:`preprocessing capabilities of OpenVINO API 2.0 <openvino_2_0_preprocessing>`
- Plugin libraries with clear names:
 - ``openvino_intel_cpu_plugin``
 - ``openvino_intel_gpu_plugin``
 - ``openvino_auto_plugin``
 - others
- Plugins to read and convert models:
 - ``openvino_ir_frontend`` to read OpenVINO IR
 - ``openvino_onnx_frontend`` to read ONNX models
 - ``openvino_paddle_frontend`` to read Paddle models
---->

@endsphinxdirective
