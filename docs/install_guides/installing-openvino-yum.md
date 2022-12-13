# Install OpenVINO™ Runtime on Linux Using YUM Repository {#openvino_docs_install_guides_installing_openvino_yum}

This guide provides installation steps for OpenVINO™ Runtime for Linux distributed through the YUM repository.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf).

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Red Hat Enterprise Linux 8 x86, 64-bit

.. tab:: Hardware

  Optimized for these processors:

  * 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors
  * 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
  * Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
  * Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
  * Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
  * Intel® Iris® Xe MAX Graphics

.. tab:: Processor Notes

  Processor graphics are not included in all processors.
  See `Product Specifications`_ for information about your processor.

  .. _Product Specifications: https://ark.intel.com/

.. tab:: Software

  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 8.2.0
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

@endsphinxdirective

## Install OpenVINO Runtime

### Step 1: Set Up the Repository

1. Create the YUM repo file in the `/tmp` directory as a normal user:
   ```
   tee > /tmp/openvino-2022.repo << EOF
   [OpenVINO]
   name=Intel(R) Distribution of OpenVINO 2022
   baseurl=https://yum.repos.intel.com/openvino/2022
   enabled=1
   gpgcheck=1
   repo_gpgcheck=1
   gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
   EOF
   ```
2.	Move the new openvino-2022.repo file to the YUM configuration directory `/etc/yum.repos.d`:
   ```sh
   sudo mv /tmp/openvino-2022.repo /etc/yum.repos.d
   ```
3.	Verify that the new repo is properly setup by running the following command:
   ```sh
   yum repolist | grep -i openvino
   ```
    You will see the available list of packages.


To list available OpenVINO packages, use the following command:

@sphinxdirective

   .. code-block:: sh

      yum list 'openvino*'

@endsphinxdirective

### Step 2: Install OpenVINO Runtime Using the YUM Package Manager

#### Install OpenVINO Runtime

@sphinxdirective

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo yum install openvino

.. tab::  A Specific Version

   Run the following command:

   .. code-block:: sh

      sudo yum install openvino-<VERSION>.<UPDATE>.<PATCH>

   For example:

   .. code-block:: sh

      sudo yum install openvino-2022.3.0

@endsphinxdirective


#### Check for Installed Packages and Version

Run the following command:

@sphinxdirective

   .. code-block:: sh

      yum list installed 'openvino*'

@endsphinxdirective


#### Uninstall OpenVINO Runtime

@sphinxdirective

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo yum autoremove openvino


.. tab::  A Specific Version

   Run the following command:

   .. code-block:: sh

      sudo yum autoremove openvino-<VERSION>.<UPDATE>.<PATCH>

   For example:

   .. code-block:: sh

      sudo yum autoremove openvino-2022.3.0

@endsphinxdirective


### Step 3 (Optional): Install Software Dependencies

After you have installed OpenVINO Runtime, if you decided to [install OpenVINO Model Development Tools](installing-model-dev-tools.md), make sure that you install external software dependencies first.

Refer to <a href="openvino_docs_install_guides_installing_openvino_linux.html#install-external-dependencies">Install External Software Dependencies</a> for detailed steps.

### Step 4 (Optional): Configure Inference on Non-CPU Devices

To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in [GPU Setup Guide](@ref openvino_docs_install_guides_configurations_for_intel_gpu).

### Step 5: Build Samples

To build the C++ or C sample applications for Linux, run the `build_samples.sh` script:

@sphinxdirective

.. tab:: C++

   .. code-block:: sh

      /usr/share/openvino/samples/cpp/build_samples.sh

.. tab:: C

   .. code-block:: sh

      /usr/share/openvino/samples/c/build_samples.sh

@endsphinxdirective

For more information, refer to <a href="openvino_docs_OV_UG_Samples_Overview.html#build-samples-linux">Build the Sample Applications on Linux</a>.

## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/nightly/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- OpenVINO™ home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).