Install OpenVINO™ Runtime via Homebrew
========================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Linux and macOS
                 operating systems, using Homebrew.

.. note::
    Recent OpenVINO releases currently unavailable via Homebrew
    Due to an issue with protobuf in the internal Homebrew CI infrastructure, the latest OpenVINO releases are temporarily unavailable via Homebrew. 
    The problem affects the ability to build and publish formula updates reliably.
    We are actively monitoring the upstream resolution and will restore availability as soon as the issue is resolved.
     
    In the meantime, we recommend using the S3 archive as an alternative to Homebrew installation on :doc:`Linux <install-openvino-archive-linux>` and :doc:`macOS <install-openvino-archive-macos>`.

.. note::

   Note that the `Homebrew <https://brew.sh/>`__ distribution:

   * offers both C/C++ and Python APIs
   * does not offer support for NPU inference
   * is dedicated to macOS (both arm64 and x86_64) and Linux (x86_64 only) users.

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

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
:doc:`list of additional configurations <./configurations>`
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
* See pre-trained deep learning models on `Hugging Face <https://huggingface.co/OpenVINO>`__.
* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../get-started/learn-openvino/openvino-samples>`.
* Check out the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__.



