# Install OpenVINO™ Runtime from Conan Package Manager {#openvino_docs_install_guides_installing_openvino_conan}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and 
                 macOS operating systems, using Conan Package Manager.

Conan Package Manager is a free, open-source dependency and package manager for C and C++ languages. 
Its main purpose is to improve development and integration of C and C++ projects. OpenVINO™ is included  
in the Conan Center Index (registry with the software packages).

Installing OpenVINO Runtime with Conan Package Manager
############################################################

1. Install Conan 2.0 or higher
   
   .. code-block:: sh

      python3 -m pip install conan

2. Create a conanfile.txt file for your OpenVINO project and add "openvino" dependency in there:

   
   .. code-block:: console

      [requires]
      openvino/2023.1.0
      [generators]
      CMakeDeps
      CMakeToolchain
      [layout]
      cmake_layout

   Run below command to create conan_toolchain.cmake file, which will be used to compile your project with OpenVINO.
   
   .. code-block:: sh

      conan install conanfile.txt --build=missing

.. note::

   By default, OpenVINO is statically compiled and all available plugins, frontends are compiled as well. You can build a tailored OpenVINO by using command below:
   
      .. code-block:: sh

         conan install conanfile.txt --build=missing -o:h openvino/*:enable_intel_gpu=False -o:h openvino/*:enable_onnx_frontend=False' -o:h openvino/*:shared=True.
   
   For more details on available options, see the `Conan Package Manager page on OpenVINO <https://conan.io/center/recipes/openvino>`__

3. Configure and compile your project with OpenVINO:

   
   .. code-block:: sh

      cmake -DCMAKE_TOOLCHAIN_FILE=<path to conan_toolchain.cmake> -DCMAKE_BUILD_TYPE=Release -S <path to CMakeLists.txt of your project> -B <build dir>
      cmake --build <build dir> --parallel

.. note::

   OpenVINO can be used with any build interface, as long as it is supported by Conan 2.0.

Additional Resources
########################

* `Conan Package Manager Webstie <https://conan.io/>`__
* :doc:`Install OpenVINO Overview <openvino_docs_install_guides_overview>`
