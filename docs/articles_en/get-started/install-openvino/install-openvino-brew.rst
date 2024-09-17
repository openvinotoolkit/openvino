Install OpenVINO™ Runtime via Homebrew
========================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Linux and macOS
                 operating systems, using Homebrew.

.. note::

   Note that the `Homebrew <https://brew.sh/>`__ distribution:

   * offers both C/C++ and Python APIs
   * does not offer support for NPU inference
   * is dedicated to macOS (both arm64 and x86_64) and Linux (x86_64 only) users.


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

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
            * GCC 7.5.0 (for Ubuntu 18.04), GCC 9.3.0 (for Ubuntu 20.04) or GCC 11.3.0 (for Ubuntu 22.04)
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`__

         .. tab-item:: macOS
            :sync: macos

            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher <https://cmake.org/download/>`__ (choose "macOS 10.13 or later"). Add ``/Applications/CMake.app/Contents/bin`` to path (for default installation).
            * `Python 3.9 - 3.12 <https://www.python.org/downloads/mac-osx/>`__ . Install and add it to path.
            * Apple Xcode Command Line Tools. In the terminal, run ``xcode-select --install`` from any directory to install it.
            * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)


Installing OpenVINO Runtime
###########################

1. Make sure that you have installed Homebrew on your system. If not, follow the instructions on `the Homebrew website <https://brew.sh/>`__ to install and configure it.

2. Run the following command in the terminal:

   .. code-block:: sh

      brew install openvino

3. Check if the installation was successful by listing all Homebrew packages:

   .. code-block:: sh

      brew list


Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <../configurations>`
to see if your case needs any of them.



Uninstalling OpenVINO
#####################

To uninstall OpenVINO via Homebrew, use the following command:

.. code-block:: sh

   brew uninstall openvino


What's Next?
####################

Now that you've installed OpenVINO Runtime, you can try the following things:

* Learn more about :doc:`OpenVINO Workflow <../../../openvino-workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <../../../openvino-workflow/model-preparation>`.
* See pre-trained deep learning models in our
  :doc:`Open Model Zoo <../../../documentation/legacy-features/model-zoo>`.

  .. important::

     Due to the deprecation of Open Model Zoo, models in the OpenVINO IR format are now
     published on `Hugging Face <https://huggingface.co/OpenVINO>`__.

* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../learn-openvino/openvino-samples>`.
* Check out the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__.



