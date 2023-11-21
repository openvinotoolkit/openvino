# Configurations for Intel® NPU with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_npu}

@sphinxdirective

.. meta::
   :description: Learn how to provide additional configuration for Intel® 
                 NPU to work with the OpenVINO™ toolkit on your system.



Drivers and Dependencies
########################


The Intel® NPU device requires a proper driver to be installed on the system.



Linux
####################

Prerequisites
++++++++++++++++++++

Ensure that make, gcc, and Linux kernel headers are installed. Use the following command to install the required software:

.. code-block:: sh

   sudo apt-get install gcc make linux-headers-generic


Configuration steps
++++++++++++++++++++











Windows
####################

Intel® NPU driver for Windows is available through Windows Update.




What’s Next?
####################

Now you are ready to try out OpenVINO™. You can use the following tutorials to write your applications using Python and C/C++.

* Developing in Python:

  * `Start with tensorflow models with OpenVINO™ <notebooks/101-tensorflow-to-openvino-with-output.html>`__
  * `Start with ONNX and PyTorch models with OpenVINO™ <notebooks/102-pytorch-onnx-to-openvino-with-output.html>`__
  * `Start with PaddlePaddle models with OpenVINO™ <notebooks/103-paddle-to-openvino-classification-with-output.html>`__

* Developing in C/C++:

  * :doc:`Image Classification Async C++ Sample <openvino_inference_engine_samples_classification_sample_async_README>`
  * :doc:`Hello Classification C++ Sample <openvino_inference_engine_samples_hello_classification_README>`
  * :doc:`Hello Reshape SSD C++ Sample <openvino_inference_engine_samples_hello_reshape_ssd_README>`

@endsphinxdirective

