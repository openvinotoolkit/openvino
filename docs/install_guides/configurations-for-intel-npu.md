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

* Supported operating systems: Ubuntu 20.04 / 22.04 LTS

* Ensure that make, gcc, and Linux kernel headers are installed. Use the following command to install the required software:

.. code-block:: sh

   sudo apt-get install gcc make linux-headers-generic


Manual Configuration steps
++++++++++++++++++++++++++

(i) After downloading the latest driver, you can unpack it with below command. Please substitute the relevant release version number:

    `tar -xf vpu-linux-drivers-ubuntu2204-release-<version number>.tar.gz`

(ii) To execute installer:

    `./vpu-linux-drivers-ubuntu2204-release-<version number>/vpu-drv-installer`

(iii) A successful installation will print the following output:

    `
    vpu-drv-installer:INFO: Removing previous installed VPU Linux Driver files
    vpu-drv-installer:INFO: VPU Linux Drivers package installed successfully
    `

(iv) To check NPU state, type:

    `dmesg`

Successful bootup of the NPU should print following message or similar:

    `[  797.193201] [drm] Initialized intel_vpu 0.<version number> for 0000:00:0b.0 on minor 0`


Windows
####################

Intel® NPU driver for Windows is available through Windows Update.

Manual Configuration steps
++++++++++++++++++++++++++

After downloading the driver, you can unpack it to a temporary folder.

(i) Click the Start Menu button → Run -> Device Manager
    This will launch Device Manager and list all the devices connected to your PC.

(ii)
a. Scan the list to see if you have the 'Intel(R) NPU Accelerator'. This means you have a previous version of the driver installed.

or

b. If you cannot find this device, then search for 'Other devices' -> 'Multimedia Video Controller'

(iii) Right click on the device you have found and select 'Update driver'

(iv) Click 'Browse my computer for drivers'
    Browse to the location of the `<unpacked driver location>\drivers\x64` folder you copied earlier, and click Next:

(v) You should get a Success message upon driver installation. Click Close.



What’s Next?
####################

Now you are ready to try out OpenVINO™. You can use the following tutorials to write your applications using Python and C++.

* Developing in Python:

  * `Start with tensorflow models with OpenVINO™ <notebooks/101-tensorflow-to-openvino-with-output.html>`__
  * `Start with ONNX and PyTorch models with OpenVINO™ <notebooks/102-pytorch-onnx-to-openvino-with-output.html>`__
  * `Start with PaddlePaddle models with OpenVINO™ <notebooks/103-paddle-to-openvino-classification-with-output.html>`__

* Developing in C++:

  * :doc:`Image Classification Async C++ Sample <openvino_inference_engine_samples_classification_sample_async_README>`
  * :doc:`Hello Classification C++ Sample <openvino_inference_engine_samples_hello_classification_README>`
  * :doc:`Hello Reshape SSD C++ Sample <openvino_inference_engine_samples_hello_reshape_ssd_README>`

@endsphinxdirective

