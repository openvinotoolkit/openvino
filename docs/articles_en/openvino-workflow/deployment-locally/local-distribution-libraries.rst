Libraries for Local Distribution
================================


.. meta::
   :description: A local distribution will have its own copies of OpenVINO
                 Runtime binaries along with a set of required libraries
                 needed to deploy the application.


With local distribution, each C or C++ application/installer has its own copies of OpenVINO Runtime binaries.
However, OpenVINO has a scalable plugin-based architecture, which means that some components
can be loaded in runtime only when they are really needed. This guide helps you understand
what minimal set of libraries is required to deploy the application.

Local distribution is also suitable for OpenVINO binaries built from source using
`Build instructions <https://github.com/openvinotoolkit/openvino/wiki#how-to-build>`__,
but this guide assumes that OpenVINO Runtime is built dynamically.
For `Static OpenVINO Runtime <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__,
select the required OpenVINO capabilities at the CMake configuration stage using
`CMake Options for Custom Compilation <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md>`__,
then build and link the OpenVINO components to the final application.

.. note::

   The steps below are independent of the operating system and refer to the library file name
   without any prefixes (like ``lib`` on Unix systems) or suffixes (like ``.dll`` on Windows OS).
   Do not put ``.lib`` files on Windows OS to the distribution because such files are needed
   only at a linker stage.


Library Requirements for C++ and C Languages
############################################

Regardless of the programming language of an application, the ``openvino`` library must always
be included in its final distribution. This core library manages all inference and frontend plugins.
The ``openvino`` library depends on the TBB libraries which are used by OpenVINO Runtime
to optimally saturate devices with computations.

If your application is in C language, you need to additionally include the ``openvino_c`` library.

Libraries for Pluggable Components
##################################

The picture below presents dependencies between the OpenVINO Runtime core and pluggable libraries:

.. image:: ../../assets/images/deployment_full.svg

Libraries for Compute Devices
+++++++++++++++++++++++++++++

For each inference device, OpenVINO Runtime has its own plugin library:

- ``openvino_intel_cpu_plugin`` for :doc:`Intel® CPU devices <../running-inference/inference-devices-and-modes/cpu-device>`
- ``openvino_intel_gpu_plugin`` for :doc:`Intel® GPU devices <../running-inference/inference-devices-and-modes/gpu-device>`
- ``openvino_intel_npu_plugin`` for :doc:`Intel® NPU devices <../running-inference/inference-devices-and-modes/npu-device>`
- ``openvino_arm_cpu_plugin`` for :doc:`ARM CPU devices <../running-inference/inference-devices-and-modes/cpu-device>`

Depending on which devices are used in the app, the corresponding libraries should be included in the distribution package.

As shown in the picture above, some plugin libraries may have OS-specific dependencies
which are either backend libraries or additional supports files with firmware, etc.
Refer to the table below for details:

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
      |              | |                       | | ``.\runtime\bin\intel64\Debug\cache.json``          |
      +--------------+-------------------------+-------------------------------------------------------+
      |     NPU      |            —            |                          —                            |
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
      |     NPU      |            —            |                          —                            |
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

The ``HETERO``, ``BATCH``, and ``AUTO`` execution modes can also be used by the application explicitly or implicitly.
Use the following recommendation scheme to decide whether to add the appropriate libraries to the distribution package:

- If :doc:`AUTO <../running-inference/inference-devices-and-modes/auto-device-selection>` is used
  explicitly in the application or ``ov::Core::compile_model`` is used without specifying a device, put ``openvino_auto_plugin`` to the distribution.

  .. note::

     Automatic Device Selection relies on :doc:`inference device plugins <../running-inference/inference-devices-and-modes>`.
     If you are not sure which inference devices are available on the target system, put all inference plugin libraries in the distribution.
     If ov::device::priorities is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If :doc:`HETERO <../running-inference/inference-devices-and-modes/hetero-execution>` is either
  used explicitly or ``ov::hint::performance_mode`` is used with GPU, put ``openvino_hetero_plugin`` in the distribution.
- If :doc:`BATCH <../running-inference/inference-devices-and-modes/automatic-batching>` is either
  used explicitly or ``ov::hint::performance_mode`` is used with GPU, put ``openvino_batch_plugin`` in the distribution.

Frontend Libraries for Reading Models
+++++++++++++++++++++++++++++++++++++

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:

- ``openvino_ir_frontend`` is used to read OpenVINO IR.
- ``openvino_tensorflow_frontend`` is used to read the TensorFlow file format.
- ``openvino_tensorflow_lite_frontend`` is used to read the TensorFlow Lite file format.
- ``openvino_onnx_frontend`` is used to read the ONNX file format.
- ``openvino_paddle_frontend`` is used to read the Paddle file format.
- ``openvino_pytorch_frontend`` is used to convert PyTorch model via ``openvino.convert_model`` API.

Depending on the model format types that are used in the application in ``ov::Core::read_model``, select the appropriate libraries.

.. note::

   To optimize the size of the final distribution package, it is recommended to convert models
   to OpenVINO IR by using :doc:`model conversion API <../model-preparation>`. This way you
   do not have to keep TensorFlow, TensorFlow Lite, ONNX, PaddlePaddle, and other frontend
   libraries in the distribution package.

Examples
####################

.. dropdown:: CPU + OpenVINO IR in C application

   In this example, the application is written in C, performs inference on CPU, and reads models stored in the OpenVINO IR format.

   The following libraries are used: ``openvino_c``, ``openvino``, ``openvino_intel_cpu_plugin``, and ``openvino_ir_frontend``.

   - The ``openvino_c`` library is a main dependency of the application. The app links against this library.
   - The ``openvino`` library is used as a private dependency for ``openvino_c`` and is also used in the deployment.
   - ``openvino_intel_cpu_plugin`` is used for inference.
   - ``openvino_ir_frontend`` is used to read source models.

.. dropdown:: Auto-Device Selection between GPU and CPU

   In this example, the application is written in C++, performs inference
   with the :doc:`Automatic Device Selection <../running-inference/inference-devices-and-modes/auto-device-selection>`
   mode, limiting device list to GPU and CPU, and reads models
   :doc:`created using C++ code <../running-inference/integrate-openvino-with-your-application/model-representation>`.

   The following libraries are used: ``openvino``, ``openvino_auto_plugin``, ``openvino_intel_gpu_plugin``, and ``openvino_intel_cpu_plugin``.

   - The ``openvino`` library is a main dependency of the application. The app links against this library.
   - ``openvino_auto_plugin`` is used to enable Automatic Device Selection.
   - ``openvino_intel_gpu_plugin`` and ``openvino_intel_cpu_plugin`` are used for inference. AUTO
     selects between CPU and GPU devices according to their physical existence on the deployed machine.
   - No frontend library is needed because ``ov::Model`` is created in code.

