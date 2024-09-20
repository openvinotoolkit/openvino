Install OpenVINO™ Runtime from Conda Forge
============================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and
                 macOS operating systems, using Conda Forge.


.. note::

   Note that the Conda Forge distribution:

   * offers both C/C++ and Python APIs
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


   .. tab-item:: Software
      :sync: software

      | There are many ways to work with Conda. Before you proceed, learn more about it on the
      | `Anaconda distribution page <https://www.anaconda.com/products/individual/>`__


Installing OpenVINO Runtime with Anaconda Package Manager
############################################################

1. Set up the Anaconda environment (Python 3.10 used as an example):

   .. code-block:: sh

      conda create --name py310 python=3.10

   .. code-block:: sh

      conda activate py310

2. Update it to the latest version:

   .. code-block:: sh

      conda update --all

3. Install the OpenVINO Runtime package:

   .. code-block:: sh

      conda install -c conda-forge openvino=2024.4.0

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the description below, as well as the
:doc:`list of additional configurations <../configurations>`
to see if your case needs any of them.

Enabling GPU device for inference
+++++++++++++++++++++++++++++++++

To use a GPU device for OpenVINO inference on Linux, you must install OpenCL ICD:

.. code-block:: sh

   conda install ocl-icd-system

This step is not required on Windows, as Intel® Graphics Compute Runtime for
OpenCL™ Driver is included with the Intel® Graphics Driver package.

Compiling with OpenVINO Runtime from Conda-Forge on Linux
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When linking OpenVINO libraries from Conda on Linux, ensure that you have the necessary Conda compilers installed and Conda standard libraries are used.
To do so, run the following command in your Conda environment:

.. code-block:: sh

    conda install cmake c-compiler cxx-compiler make
    conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

It is crucial to reactivate your Conda environment after installing the compilers.
This step ensures that all the environment variables are set correctly for successful linkage.

To reactivate your Conda environment, execute the following command:

.. code-block:: sh

    conda activate py310

Once you have reactivated your Conda environment, make sure that all the necessary environment
variables are properly set and proceed with linking the OpenVINO libraries.

Installing specific components of OpenVINO from Conda Forge
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You do not have to install the entire OpenVINO package. You can install selected
components by using:

.. code-block:: sh

   conda install conda-forge::<component_name>

``<component_name>`` may be one of the components of OpenVINO listed below:

- ``libopenvino-auto-batch-plugin``
- ``libopenvino-auto-plugin``
- ``libopenvino-hetero-plugin``
- ``libopenvino-intel-cpu-plugin``
- ``libopenvino-intel-gpu-plugin``
- ``libopenvino-intel-npu-plugin``
- ``libopenvino-ir-frontend``
- ``libopenvino-onnx-frontend``
- ``libopenvino-paddle-frontend``
- ``libopenvino-pytorch-frontend``
- ``libopenvino-tensorflow-frontend``
- ``libopenvino-tensorflow-lite-frontend``
- ``libopenvino-dev``
- ``libopenvino-python``
- ``libopenvino-arm-cpu-plugin``


For example, to install a single component, use:

.. code-block:: sh

   conda install conda-forge::libopenvino-intel-cpu-plugin

For multiple components, use:

.. code-block:: sh

   conda install conda-forge::libopenvino-intel-cpu-plugin conda-forge::libopenvino-arm-cpu-plugin conda-forge::libopenvino-intel-npu-plugin conda-forge::libopenvino-intel-gpu-plugin

Uninstalling OpenVINO™ Runtime
###########################################################

Once OpenVINO Runtime is installed via Conda, you can remove it, using the following command,
with the proper OpenVINO version number:

.. code-block:: sh

   conda remove openvino=2024.4.0

If you have installed specific components of OpenVINO, you can remove them, using:

.. code-block:: sh

   conda remove conda-forge::<component_name>

What's Next?
############################################################

Now that you've installed OpenVINO Runtime, you are ready to run your own machine learning applications!
To learn more about how to integrate a model in OpenVINO applications, try out some tutorials and sample applications.

Try the :doc:`C++ Quick Start Example <../../../learn-openvino/openvino-samples/get-started-demos>` for step-by-step instructions
on building and running a basic image classification C++ application.

.. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
   :width: 400

Visit the :doc:`Samples <../../../learn-openvino/openvino-samples>` page for other C++ example applications to get you started with OpenVINO, such as:

* :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../learn-openvino/openvino-samples/hello-reshape-ssd>`
* :doc:`Object classification sample <../../../learn-openvino/openvino-samples/hello-classification>`

