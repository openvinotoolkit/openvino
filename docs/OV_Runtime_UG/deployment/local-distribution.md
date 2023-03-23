# Libraries for Local Distribution {#openvino_docs_deploy_local_distribution}

@sphinxdirective

With a local distribution, each C or C++ application/installer will have its own copies of OpenVINO Runtime binaries. However, OpenVINO has a scalable plugin-based architecture, which means that some components can be loaded in runtime only when they are really needed. Therefore, it is important to understand which minimal set of libraries is really needed to deploy the application. This guide helps you to achieve that goal.

Local distribution is also appropriate for OpenVINO binaries built from sources using `Build instructions <https://github.com/openvinotoolkit/openvino/wiki#how-to-build>`__, 
but the guide below supposes OpenVINO Runtime is built dynamically. For case of `Static OpenVINO Runtime <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__ select the required OpenVINO capabilities on CMake configuration stage using `CMake Options for Custom Compilation <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_comiplation.md>`__, the build and link the OpenVINO components into the final application.

.. note::

   The steps below are operating system independent and refer to a library file name without any prefixes (like ``lib`` on Unix systems) or suffixes (like ``.dll`` on Windows OS). Do not put ``.lib`` files on Windows OS to the distribution, because such files are needed only on a linker stage.


Library Requirements for C++ and C Languages
############################################

Independent on the language used to write the application, the ``openvino`` library must always be put to the final distribution, since it's a core library which orchestrates with all the inference and frontend plugins. In Intel® Distribution of OpenVINO™ toolkit, ``openvino`` depends on the TBB libraries which are used by OpenVINO Runtime to optimally saturate the devices with computations, so it must be put to the distribution package.

If your application is written with C language, you need to put the ``openvino_c`` library additionally.

The ``plugins.xml`` file with information about inference devices must also be taken as a support file for ``openvino``.


Libraries for Pluggable Components
##################################

The picture below presents dependencies between the OpenVINO Runtime core and pluggable libraries:

.. image:: _static/images/deployment_full.svg

Libraries for Compute Devices
+++++++++++++++++++++++++++++

For each inference device, OpenVINO Runtime has its own plugin library:

- ``openvino_intel_cpu_plugin`` for :doc:`Intel® CPU devices <openvino_docs_OV_UG_supported_plugins_CPU>`.
- ``openvino_intel_gpu_plugin`` for :doc:`Intel® GPU devices <openvino_docs_OV_UG_supported_plugins_GPU>`.
- ``openvino_intel_gna_plugin`` for :doc:`Intel® GNA devices <openvino_docs_OV_UG_supported_plugins_GNA>`.
- ``openvino_arm_cpu_plugin`` for :doc:`ARM CPU devices <openvino_docs_OV_UG_supported_plugins_ARM_CPU>`.

Depending on what devices are used in the app, the appropriate libraries need to be put to the distribution package.

As it is shown on the picture above, some plugin libraries may have OS-specific dependencies which are either backend libraries or additional supports files with firmware, etc. Refer to the table below for details:

.. dropdown:: Windows OS:

   .. list-table::
      :header-rows: 1

      * - Device
        - Dependency
      * - CPU
        - ``-``
      * - GPU
        - ``OpenCL.dll``, ``cache.json``
      * - GNA
        - ``gna.dll``
      * - Arm® CPU
        - ``-``


.. dropdown:: Linux OS:

   .. list-table::
      :header-rows: 1

      * - Device
        - Dependency
      * - CPU
        - ``-``
      * - GPU
        - ``libOpenCL.so``, ``cache.json``
      * - GNA
        - ``gna.dll``
      * - Arm® CPU
        - ``-``


.. dropdown:: MacOS:

   .. list-table::
      :header-rows: 1

      * - Device
        - Dependency
      * - CPU
        - ``-``
      * - Arm® CPU
        - ``-``


Libraries for Execution Modes
+++++++++++++++++++++++++++++

The ``HETERO``, ``MULTI``, ``BATCH`` and ``AUTO`` execution modes can also be used explicitly or implicitly by the application. Use the following recommendation scheme to decide whether to put the appropriate libraries to the distribution package:

- If :doc:`AUTO <openvino_docs_OV_UG_supported_plugins_AUTO>` is used explicitly in the application or `ov::Core::compile_model <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__ is used without specifying a device, put ``openvino_auto_plugin`` to the distribution.

  .. note::

     Automatic Device Selection relies on :doc:`[inference device plugins <openvino_docs_OV_UG_Working_with_devices>`. If you are not sure about what inference devices are available on target system, put all the inference plugin libraries to the distribution. If `ov::device::priorities <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gae88af90a18871677f39739cb0ef0101e>`__ is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If :doc:`MULTI <openvino_docs_OV_UG_Running_on_multiple_devices>` is used explicitly, put ``openvino_auto_plugin`` to the distribution.
- If :doc:`HETERO <openvino_docs_OV_UG_Hetero_execution>` is either used explicitly or `ov::hint::performance_mode <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1ga2691fe27acc8aa1d1700ad40b6da3ba2>`__ is used with GPU, put ``openvino_hetero_plugin`` to the distribution.
- If :doc:`BATCH <openvino_docs_OV_UG_Automatic_Batching>` is either used explicitly or ``ov::hint::performance_mode`` is used with GPU, put ``openvino_batch_plugin`` to the distribution.

Frontend Libraries for Reading Models
+++++++++++++++++++++++++++++++++++++

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:

- ``openvino_ir_frontend`` is used to read OpenVINO IR.
- ``openvino_tensorflow_frontend`` is used to read TensorFlow file format.
- ``openvino_onnx_frontend`` is used to read ONNX file format.
- ``openvino_paddle_frontend`` is used to read Paddle file format.

Depending on the model format types that are used in the application in `ov::Core::read_model <classov_1_1Core.html#doxid-classov-1-1-core-1ae0576a95f841c3a6f5e46e4802716981>`__, pick up the appropriate libraries.

.. note::

   To optimize the size of final distribution package, you are recommended to convert models to OpenVINO IR by using :doc:`Model Optimizer <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`. This way you don't have to keep TensorFlow, ONNX, PaddlePaddle, and other frontend libraries in the distribution package.

(Legacy) Preprocessing via G-API
++++++++++++++++++++++++++++++++

.. note::

   :doc:`G-API <openvino_docs_gapi_gapi_intro>` preprocessing is a legacy functionality, use :doc:`preprocessing capabilities from OpenVINO 2.0 <openvino_docs_OV_UG_Preprocessing_Overview>` which do not require any additional libraries.

If the application uses `InferenceEngine::PreProcessInfo::setColorFormat <classInferenceEngine_1_1PreProcessInfo.html#doxid-class-inference-engine-1-1-pre-process-info-1a3a10ba0d562a2268fe584d4d2db94cac>`__ or `InferenceEngine::PreProcessInfo::setResizeAlgorithm <classInferenceEngine_1_1PreProcessInfo.html#doxid-class-inference-engine-1-1-pre-process-info-1a0c083c43d01c53c327f09095e3e3f004>`__ methods, OpenVINO Runtime dynamically loads `openvino_gapi_preproc` plugin to perform preprocessing via G-API.

Examples
####################

**CPU + OpenVINO IR in C application**

In this example, the application is written in C language, performs inference on CPU, and reads models stored as the OpenVINO IR format. The following libraries are used:
- The ``openvino_c`` library is a main dependency of the application. It links against this library.
- The ``openvino`` library is used as a private dependency for ``openvino_c`` and is also used in the deployment.
- ``openvino_intel_cpu_plugin`` is used for inference.
- ``openvino_ir_frontend`` is used to read source models.

**MULTI execution on GPU and CPU in `tput` mode**

In this example, the application is written in C++, performs inference :doc:`simultaneously on GPU and CPU devices <openvino_docs_OV_UG_Running_on_multiple_devices>` with the `ov::hint::PerformanceMode::THROUGHPUT <enumov_1_1hint_1_1PerformanceMode.html#doxid-group-ov-runtime-cpp-prop-api-1gga032aa530efa40760b79af14913d48d73a50f9b1f40c078d242af7ec323ace44b3>`__ property set, and reads models stored in the ONNX format. The following libraries are used:

- The ``openvino`` library is a main dependency of the application. It links against this library.
- ``openvino_intel_gpu_plugin`` and ``openvino_intel_cpu_plugin`` are used for inference.
- ``openvino_auto_plugin`` is used for Multi-Device Execution.
- ``openvino_auto_batch_plugin`` can be also put to the distribution to improve the saturation of :doc:`Intel® GPU <openvino_docs_OV_UG_supported_plugins_GPU>` device. If there is no such plugin, :doc:`Automatic Batching <openvino_docs_OV_UG_Automatic_Batching>` is turned off.
- ``openvino_onnx_frontend`` is used to read source models.

**Auto-Device Selection between GPU and CPU**

In this example, the application is written in C++, performs inference with the :doc:`Automatic Device Selection <openvino_docs_OV_UG_supported_plugins_AUTO>` mode, limiting device list to GPU and CPU, and reads models :doc:`created using C++ code <openvino_docs_OV_UG_Model_Representation>`. The following libraries are used:

- The ``openvino`` library is a main dependency of the application. It links against this library.
- ``openvino_auto_plugin`` is used to enable Automatic Device Selection.
- ``openvino_intel_gpu_plugin`` and ``openvino_intel_cpu_plugin`` are used for inference. AUTO selects between CPU and GPU devices according to their physical existence on the deployed machine.
- No frontend library is needed because ``ov::Model`` is created in code.

@endsphinxdirective
