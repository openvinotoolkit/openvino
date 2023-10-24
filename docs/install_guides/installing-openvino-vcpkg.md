# Install OpenVINO™ Runtime via vcpkg {#openvino_docs_install_guides_installing_openvino_vcpkg}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and macOS 
                 operating systems, using vcpkg.

.. note::
   
   Note that the vcpkg distribution:

   * offers C/C++ API only
   * does not offer support for GNA and NPU inference
   * is dedicated to users of all major OSes: Windows, Linux, and macOS 
     (all x86_64 / arm64 architectures)

.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <system_requirements>`
   
   .. tab-item:: Processor Notes
      :sync: processor-notes
   
      | To see if your processor includes the integrated graphics technology and supports iGPU inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__

   .. tab-item:: Software Requirements
      :sync: software-requirements

      * `vcpkg <https://vcpkg.io/en/getting-started>`__



Installing OpenVINO Runtime
###########################

1. Make sure that you have installed vcpkg on your system. If not, follow the 
   `vcpkg installation instructions <https://vcpkg.io/en/getting-started>`__.


2. Install OpenVINO using the following terminal command:

   .. code-block:: sh

      vcpkg install openvino

   vcpkg also enables you to install only selected components, by specifying them in the command.
   See the list of `available features <https://vcpkg.link/ports/openvino>`__, for example: 

   .. code-block:: sh

      vcpkg install 'openvino[core,cpu,ir]'

   vcpkg also provides a way to install OpenVINO for any specific configuration you want via `triplets <https://learn.microsoft.com/en-us/vcpkg/users/triplets>`__, for example to install OpenVINO statically on Windows, use:

   .. code-block:: sh

      vcpkg install openvino:x64-windows-static

Note that the vcpkg installation means building all packages and dependencies from source, 
which means the compiler stage will require additional time to complete the process. 

.. important::

   If you are building OpenVINO as dynamic libraries and you want to use either Paddle, TensorFlow or ONNX frontends, you need to create `custom vcpkg <https://learn.microsoft.com/en-us/vcpkg/users/triplets#per-port-customization>`__ triplet file, like ``<VCPKG_ROOT>/triplets/community/x64-linux-release-dynamic.cmake``, which builds ``protobuf`` dependency statically:

   .. code-block:: sh

      # typical values of vcpkg toolchain
      set(VCPKG_TARGET_ARCHITECTURE x64)
      set(VCPKG_CRT_LINKAGE dynamic)
      # by default, all libraries are built dynamically
      set(VCPKG_LIBRARY_LINKAGE dynamic)

      set(VCPKG_CMAKE_SYSTEM_NAME Linux)
      set(VCPKG_BUILD_TYPE release)

      set(VCPKG_FIXUP_ELF_RPATH ON)

      # OpenVINO specific additions: build statically the following internal dependencies
      # IMPORTANT: you need to build at least protobuf statically, others can be dynamic
      if(PORT MATCHES "^(ade|hwloc|onnx|protobuf|pugixml|snappy)$")
          set(VCPKG_LIBRARY_LINKAGE static)
      endif()


   Then, you can use such a triplet file with the following command:

   .. code-block:: sh

      vcpkg install openvino:x64-linux-release-dynamic


After installation, you can use OpenVINO in your product's cmake scripts:

.. code-block:: sh

   find_package(OpenVINO REQUIRED)

And running from terminal:

.. code-block:: sh

   cmake -B <build dir> -S <source dir> -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake

Congratulations! You've just Installed and used OpenVINO in your project! For some use cases you may still
need to install additional components. Check the 
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.

Uninstalling OpenVINO
#####################

To uninstall OpenVINO via vcpkg, use the following command:

.. code-block:: sh

   vcpkg uninstall openvino


What's Next?
####################

Now that you've installed OpenVINO Runtime, you can try the following things:

* Learn more about :doc:`OpenVINO Workflow <openvino_workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <openvino_docs_model_processing_introduction>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.
* Check out the OpenVINO product home page: https://software.intel.com/en-us/openvino-toolkit.



@endsphinxdirective
