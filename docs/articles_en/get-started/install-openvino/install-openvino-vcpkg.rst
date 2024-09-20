Install OpenVINO™ Runtime via vcpkg
=====================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and macOS
                 operating systems, using vcpkg.

.. note::

   Note that the vcpkg distribution:

   * offers C/C++ API only
   * does not offer support for NPU inference
   * is dedicated to users of all major OSes: Windows, Linux, and macOS
     (all x86_64 / arm64 architectures)

.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <../../../about-openvino/release-notes-openvino/system-requirements>`

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

      vcpkg install 'openvino:x64-windows-static'

Note that the vcpkg installation means building all packages and dependencies from source,
which means the compiler stage will require additional time to complete the process.

After installation, you can use OpenVINO in your product's cmake scripts:

.. code-block:: sh

   find_package(OpenVINO REQUIRED)

And running from terminal:

.. code-block:: sh

   cmake -B <build dir> -S <source dir> -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake

Congratulations! You've just Installed and used OpenVINO in your project! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <../configurations>`
to see if your case needs any of them.

Uninstalling OpenVINO
#####################

To uninstall OpenVINO via vcpkg, use the following command:

.. code-block:: sh

   vcpkg uninstall openvino


What's Next?
####################

Now that you've installed OpenVINO Runtime, you can try the following things:

* Learn more about :doc:`OpenVINO Workflow <../../../openvino-workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <../../../openvino-workflow/model-preparation>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <../../../documentation/legacy-features/model-zoo>`.

  .. important::

     Due to the deprecation of Open Model Zoo, models in the OpenVINO IR format are now
     published on `Hugging Face <https://huggingface.co/OpenVINO>`__.

* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../learn-openvino/openvino-samples>`.
* Check out the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__ .



