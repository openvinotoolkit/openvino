Install OpenVINO™ Runtime from Conan Package Manager
======================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and
                 macOS operating systems, using Conan Package Manager.

.. note::

   Note that the Conan Package Manager distribution:

   * offers C/C++ API only
   * does not offer support for NPU inference
   * is dedicated to users of all major OSes: Windows, Linux, and macOS
     (all x86_64 / arm64 architectures)

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

Installing OpenVINO Runtime with Conan Package Manager
############################################################

1. `Install Conan <https://docs.conan.io/2/installation.html>`__ 2.0.8 or higher, for example, using pip:

   .. code-block:: sh

      python3 -m pip install 'conan>=2.0.8'

2. Create a ``conanfile.txt`` file for your OpenVINO project and add "*openvino*" dependency in there:

   .. code-block:: sh

      [requires]
      openvino/2025.1.0
      [generators]
      CMakeDeps
      CMakeToolchain
      [layout]
      cmake_layout

   Run the command below to create ``conan_toolchain.cmake`` file, which will be used to compile your project with OpenVINO:

   .. code-block:: sh

      conan install conanfile.txt --build=missing

   By default, OpenVINO is statically compiled, together with all available
   plugins and frontends. To build a version tailored to your needs, check
   what options there are on the `Conan Package Manager page for OpenVINO <https://conan.io/center/recipes/openvino>`__
   and extend the command, like so:

   .. code-block:: sh

      conan install conanfile.txt --build=missing -o:h 'openvino/*:enable_intel_gpu=False' -o:h 'openvino/*:enable_onnx_frontend=False' -o:h 'openvino/*:shared=True'

3. Configure and compile your project with OpenVINO:

   .. code-block:: sh

      cmake -DCMAKE_TOOLCHAIN_FILE=<path to conan_toolchain.cmake> -DCMAKE_BUILD_TYPE=Release -S <path to CMakeLists.txt of your project> -B <build dir>
      cmake --build <build dir> --parallel

   .. note::

      OpenVINO can be used with any build interface, as long as it is supported by Conan 2.0. Read `more <https://docs.conan.io/2/examples/tools.html>`__.

Additional Resources
########################

* `Conan Package Manager <https://conan.io>`__.
* Learn more about :doc:`OpenVINO Workflow <../../../openvino-workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <../../../openvino-workflow/model-preparation>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../get-started/learn-openvino/openvino-samples>`.
* Check out the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__.


