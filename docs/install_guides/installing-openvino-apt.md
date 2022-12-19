# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

This guide provides detailed steps for installing OpenVINO™ Runtime through the APT repository and guidelines for installing OpenVINO Development Tools.

> **NOTE**: From the 2022.1 release, OpenVINO™ Development Tools can be installed via PyPI only. See [Install OpenVINO Development Tools](#installing-openvino-development-tools) for more information.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf).

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Ubuntu 18.04 long-term support (LTS) x86, 64-bit
  * Ubuntu 20.04 long-term support (LTS) x86, 64-bit

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
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

@endsphinxdirective

## Installing OpenVINO Runtime

### Step 1: Set Up the OpenVINO Toolkit APT Repository

1. Install the GPG key for the repository

    a. Download the [GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB](https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB).

        You can also use the following command:
        ```sh
        wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        ```

    b. Add this key to the system keyring:
        ```sh
        sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        ```

    > **NOTE**: You might need to install GnuPG: `sudo apt-get install gnupg`

2.	Add the repository via the following command:
@sphinxdirective

.. tab:: Ubuntu 18

   .. code-block:: sh

      echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

.. tab:: Ubuntu 20

   .. code-block:: sh

      echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

@endsphinxdirective


3.	Update the list of packages via the update command:
   ```sh
   sudo apt update
   ```

4.	Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:
   ```sh
   apt-cache search openvino
   ```


### Step 2: Install OpenVINO Runtime Using the APT Package Manager

#### Install OpenVINO Runtime

@sphinxdirective

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo apt install openvino


.. tab::  A Specific Version

   1. Get a list of OpenVINO packages available for installation:

      .. code-block:: sh

         sudo apt-cache search openvino

   2. Install a specific version of an OpenVINO package:

      .. code-block:: sh

         sudo apt install openvino-<VERSION>.<UPDATE>.<PATCH>

      For example:

      .. code-block:: sh

         sudo apt install openvino-2022.3.0

.. note::

   You can use `--no-install-recommends` option to install only required packages. Keep in mind that the build tools must be installed **separately** if you want to compile the samples.

@endsphinxdirective


#### Check for Installed Packages and Versions

Run the following command:
```sh
apt list --installed | grep openvino
```

#### Uninstall OpenVINO Runtime

@sphinxdirective

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo apt autoremove openvino


.. tab::  A Specific Version

   Run the following command:

   .. code-block:: sh

      sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>

   For example:

   .. code-block:: sh

      sudo apt autoremove openvino-2022.3.0

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

## Installing OpenVINO Development Tools

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can be installed via PyPI only.

To install OpenVINO Development Tools, do the following steps:
1. [Install OpenVINO Runtime](#installing-openvino-runtime) if you haven't done it yet.
2. <a href="openvino_docs_install_guides_installing_openvino_linux.html#install-external-dependencies">Install External Software Dependencies</a>.
3. See the **For C++ Developers** section in [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/nightly/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).