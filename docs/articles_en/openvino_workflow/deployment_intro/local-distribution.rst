.. {#openvino_docs_deploy_local_distribution}

Libraries for Local Distribution
================================


.. meta::
   :description: A local distribution will have its own copies of OpenVINO 
                 Runtime binaries along with a set of required libraries 
                 needed to deploy the application.


With local distribution, each C or C++ application/installer has its own copies of OpenVINO Runtime binaries. However, OpenVINO has a scalable plugin-based architecture, which means that some components can be loaded in runtime only when they are really needed. This guide helps you understand what minimal set of libraries is required to deploy the application.

Local distribution is also suitable for OpenVINO binaries built from source using `Build instructions <https://github.com/openvinotoolkit/openvino/wiki#how-to-build>`__, 
but this guide assumes that OpenVINO Runtime is built dynamically. For `Static OpenVINO Runtime <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__, select the required OpenVINO capabilities at the CMake configuration stage using `CMake Options for Custom Compilation <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md>`__, then build and link the OpenVINO components to the final application.

.. note::

   The steps below are independent of the operating system and refer to the library file name without any prefixes (like ``lib`` on Unix systems) or suffixes (like ``.dll`` on Windows OS). Do not put ``.lib`` files on Windows OS to the distribution because such files are needed only at a linker stage.


Library Requirements for C++ and C Languages
############################################

Regardless of the programming language of an application, the ``openvino`` library must always be included in its final distribution. This core library manages all inference and frontend plugins. The ``openvino`` library depends on the TBB libraries which are used by OpenVINO Runtime to optimally saturate devices with computations.

If your application is in C language, you need to additionally include the ``openvino_c`` library.

The ``plugins.xml`` file with information about inference devices must also be taken as a support file for ``openvino``.


Libraries for Pluggable Components
##################################

The picture below presents dependencies between the OpenVINO Runtime core and pluggable libraries:

.. image:: _static/images/deployment_full.svg

Libraries for Compute Devices
+++++++++++++++++++++++++++++

For each inference device, OpenVINO Runtime has its own plugin library:

- ``openvino_intel_cpu_plugin`` for :doc:`Intel® CPU devices <openvino_docs_OV_UG_supported_plugins_CPU>`
- ``openvino_intel_gpu_plugin`` for :doc:`Intel® GPU devices <openvino_docs_OV_UG_supported_plugins_GPU>`
- ``openvino_intel_gna_plugin`` for :doc:`Intel® GNA devices <openvino_docs_OV_UG_supported_plugins_GNA>`
- ``openvino_arm_cpu_plugin`` for :doc:`ARM CPU devices <openvino_docs_OV_UG_supported_plugins_CPU>`

Depending on which devices are used in the app, the corresponding libraries should be included in the distribution package.

As shown in the picture above, some plugin libraries may have OS-specific dependencies which are either backend libraries or additional supports files with firmware, etc. Refer to the table below for details:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      +--------------+-------------------------+-------------------------------------------------------+
      |    Device    |       Dependency        |                      Location                         |
      +==============+=========================+=======================================================+
      |     CPU      |            —            |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+
      |     GPU      | | OpenCL.dll            | | ``C:\Windows\System32\opencl.dll``                  |
      |              | | cache.json            | | ``.\runtime\bin\intel64\Release\cache.json``   or   |
      |              |                         | | ``.\runtime\bin\intel64\Debug\cache.json``          |
      +--------------+-------------------------+-------------------------------------------------------+
      |     GNA      |         gna.dll         | | ``.\runtime\bin\intel64\Release\gna.dll``    or     |
      |              |                         | | ``.\runtime\bin\intel64\Debug\gna.dll``             |
      +--------------+-------------------------+-------------------------------------------------------+
      |  Arm® CPU    |            —            |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+

   .. tab-item:: Linux arm64
      :sync: linux-arm-64

      +--------------+-------------------------+-------------------------------------------------------+
      |    Device    |       Dependency        |                      Location                         |
      +==============+=========================+=======================================================+
      |  Arm® CPU    |            —            |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+

   .. tab-item:: Linux x86_64
      :sync: linux-x86-64

      +--------------+-------------------------+-------------------------------------------------------+
      |    Device    |       Dependency        |                      Location                         |
      +==============+=========================+=======================================================+
      |     CPU      |            —            |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+
      |     GPU      | | libOpenCL.so          | | ``/usr/lib/x86_64-linux-gnu/libOpenCL.so.1``        |
      |              | | cache.json            | | ``./runtime/lib/intel64/cache.json``                |
      +--------------+-------------------------+-------------------------------------------------------+
      |     GNA      |      libgna.so          | ``./runtime/lib/intel64/libgna.so.3``                 |
      +--------------+-------------------------+-------------------------------------------------------+

   .. tab-item:: macOS arm64
      :sync: macos-arm-64

      +--------------+-------------------------+-------------------------------------------------------+
      |    Device    |       Dependency        |                      Location                         |
      +==============+=========================+=======================================================+
      |  Arm® CPU    |           —             |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+

   .. tab-item:: macOS x86_64
      :sync: macos-x86-64

      +--------------+-------------------------+-------------------------------------------------------+
      |    Device    |       Dependency        |                      Location                         |
      +==============+=========================+=======================================================+
      |     CPU      |           —             |                          —                            |
      +--------------+-------------------------+-------------------------------------------------------+



Libraries for Execution Modes
+++++++++++++++++++++++++++++

The ``HETERO``, ``MULTI``, ``BATCH`` and ``AUTO`` execution modes can also be used by the application explicitly or implicitly. Use the following recommendation scheme to decide whether to add the appropriate libraries to the distribution package:

- If :doc:`AUTO <openvino_docs_OV_UG_supported_plugins_AUTO>` is used explicitly in the application or `ov::Core::compile_model <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__ is used without specifying a device, put ``openvino_auto_plugin`` to the distribution.

  .. note::

     Automatic Device Selection relies on :doc:`inference device plugins <openvino_docs_OV_UG_Working_with_devices>`. If you are not sure which inference devices are available on the target system, put all inference plugin libraries in the distribution. If `ov::device::priorities <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gae88af90a18871677f39739cb0ef0101e>`__ is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If :doc:`MULTI <openvino_docs_OV_UG_Running_on_multiple_devices>` is used explicitly, put ``openvino_auto_plugin`` in the distribution.
- If :doc:`HETERO <openvino_docs_OV_UG_Hetero_execution>` is either used explicitly or `ov::hint::performance_mode <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1ga2691fe27acc8aa1d1700ad40b6da3ba2>`__ is used with GPU, put ``openvino_hetero_plugin`` in the distribution.
- If :doc:`BATCH <openvino_docs_OV_UG_Automatic_Batching>` is either used explicitly or ``ov::hint::performance_mode`` is used with GPU, put ``openvino_batch_plugin`` in the distribution.

Frontend Libraries for Reading Models
+++++++++++++++++++++++++++++++++++++

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:

- ``openvino_ir_frontend`` is used to read OpenVINO IR.
- ``openvino_tensorflow_frontend`` is used to read the TensorFlow file format.
- ``openvino_tensorflow_lite_frontend`` is used to read the TensorFlow Lite file format.
- ``openvino_onnx_frontend`` is used to read the ONNX file format.
- ``openvino_paddle_frontend`` is used to read the Paddle file format.
- ``openvino_pytorch_frontend`` is used to convert PyTorch model via ``openvino.convert_model`` API.

Depending on the model format types that are used in the application in `ov::Core::read_model <classov_1_1Core.html#doxid-classov-1-1-core-1ae0576a95f841c3a6f5e46e4802716981>`__, select the appropriate libraries.

.. note::

   To optimize the size of the final distribution package, it is recommended to convert models to OpenVINO IR by using :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`. This way you do not have to keep TensorFlow, TensorFlow Lite, ONNX, PaddlePaddle, and other frontend libraries in the distribution package.

Examples
####################

**CPU + OpenVINO IR in C application**

In this example, the application is written in C, performs inference on CPU, and reads models stored in the OpenVINO IR format. 

The following libraries are used: ``openvino_c``, ``openvino``, ``openvino_intel_cpu_plugin``, and ``openvino_ir_frontend``.

- The ``openvino_c`` library is a main dependency of the application. The app links against this library.
- The ``openvino`` library is used as a private dependency for ``openvino_c`` and is also used in the deployment.
- ``openvino_intel_cpu_plugin`` is used for inference.
- ``openvino_ir_frontend`` is used to read source models.

**MULTI execution on GPU and CPU in `tput` mode**

In this example, the application is written in C++, performs inference :doc:`simultaneously on GPU and CPU devices <openvino_docs_OV_UG_Running_on_multiple_devices>` with the `ov::hint::PerformanceMode::THROUGHPUT <enumov_1_1hint_1_1PerformanceMode.html#doxid-group-ov-runtime-cpp-prop-api-1gga032aa530efa40760b79af14913d48d73a50f9b1f40c078d242af7ec323ace44b3>`__ property set, and reads models stored in the ONNX format. 

The following libraries are used: ``openvino``, ``openvino_intel_gpu_plugin``, ``openvino_intel_cpu_plugin``, ``openvino_auto_plugin``, ``openvino_auto_batch_plugin``, and ``openvino_onnx_frontend``. 

- The ``openvino`` library is a main dependency of the application. The app links against this library.
- ``openvino_intel_gpu_plugin`` and ``openvino_intel_cpu_plugin`` are used for inference.
- ``openvino_auto_plugin`` is used for Multi-Device Execution.
- ``openvino_auto_batch_plugin`` can be also put in the distribution to improve the saturation of :doc:`Intel® GPU <openvino_docs_OV_UG_supported_plugins_GPU>` device. If there is no such plugin, :doc:`Automatic Batching <openvino_docs_OV_UG_Automatic_Batching>` is turned off.
- ``openvino_onnx_frontend`` is used to read source models.

**Auto-Device Selection between GPU and CPU**

In this example, the application is written in C++, performs inference with the :doc:`Automatic Device Selection <openvino_docs_OV_UG_supported_plugins_AUTO>` mode, limiting device list to GPU and CPU, and reads models :doc:`created using C++ code <openvino_docs_OV_UG_Model_Representation>`. 

The following libraries are used: ``openvino``, ``openvino_auto_plugin``, ``openvino_intel_gpu_plugin``, and ``openvino_intel_cpu_plugin``. 

- The ``openvino`` library is a main dependency of the application. The app links against this library.
- ``openvino_auto_plugin`` is used to enable Automatic Device Selection.
- ``openvino_intel_gpu_plugin`` and ``openvino_intel_cpu_plugin`` are used for inference. AUTO selects between CPU and GPU devices according to their physical existence on the deployed machine.
- No frontend library is needed because ``ov::Model`` is created in code.

