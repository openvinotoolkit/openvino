# Configurations for Intel® Gaussian & Neural Accelerator (GNA) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gna}

@sphinxdirective

.. note:: On platforms where Intel® GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

@endsphinxdirective


## Drivers and Dependencies

@sphinxdirective

Intel® GNA hardware requires a driver to be installed on the system.

.. _gna guide:

@endsphinxdirective


## Linux

### Prerequisites

Ensure that make, gcc, and Linux kernel headers are installed.

### Configuration steps

@sphinxdirective

#. Download `Intel® GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.4+) <https://storage.openvinotoolkit.org/drivers/gna/>`__
#. Run the sample_install.sh script provided in the installation package:

   .. code-block:: sh

      prompt$ ./scripts/sample_install.sh


You can also build and install the driver manually by using the following commands:

.. code-block:: sh

   prompt$ cd src/
   prompt$ make
   prompt$ sudo insmod intel_gna.ko


To unload the driver:

.. code-block:: sh

   prompt$ sudo rmmod intel_gna


.. _gna guide windows:

@endsphinxdirective


## Windows

Intel® GNA driver for Windows is available through Windows Update.

## What’s Next?

@sphinxdirective

Now you are ready to try out OpenVINO™. You can use the following tutorials to write your applications using Python and C++.

Developing in Python:
   * `Start with tensorflow models with OpenVINO™ <https://docs.openvino.ai/nightly/notebooks/101-tensorflow-to-openvino-with-output.html>`_
   * `Start with ONNX and PyTorch models with OpenVINO™ <https://docs.openvino.ai/nightly/notebooks/102-pytorch-onnx-to-openvino-with-output.html>`_
   * `Start with PaddlePaddle models with OpenVINO™ <https://docs.openvino.ai/nightly/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html>`_

Developing in C++:
   * :doc:`Image Classification Async C++ Sample <openvino_inference_engine_samples_classification_sample_async_README>`
   * :doc:`Hello Classification C++ Sample <openvino_inference_engine_samples_hello_classification_README>`
   * :doc:`Hello Reshape SSD C++ Sample <openvino_inference_engine_samples_hello_reshape_ssd_README>`

@endsphinxdirective

