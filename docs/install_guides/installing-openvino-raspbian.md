# Install OpenVINO™ toolkit for Raspbian OS {#openvino_docs_install_guides_installing_openvino_raspbian}

@sphinxdirective
.. note::
  * These steps apply to 32-bit Raspbian OS, which is an official OS for Raspberry Pi boards.
  * These steps have been validated with Raspberry Pi 3.
  * There is also an open-source version of OpenVINO™ that can be compiled for arch64 (see `build instructions <https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS>`_).
@endsphinxdirective

## Development and Target Platforms

@sphinxdirective
.. tab:: Operating Systems

  * Raspbian Buster, 32-bit
  * Raspbian Stretch, 32-bit

.. tab:: Hardware

  * Raspberry Pi board with ARM ARMv7-A CPU architecture. Check that `uname -m` returns `armv7l`.
  * Intel® Neural Compute Stick 2, which as one of the Intel® Movidius™ Visual Processing Units (VPUs)

  .. note::
    The current version of the Intel® Distribution of OpenVINO™ toolkit for Raspbian OS supports inference on Intel CPUs and Intel® Neural Compute Stick 2 devices only.

.. tab:: Software Requirements

  * CMake 3.7.2 or higher
  * Python 3.7-3.8, 32-bit

@endsphinxdirective

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit for Raspbian OS. The following steps will be covered:

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#install-external-dependencies">Install External Software Dependencies</a>
3. <a href="#set-the-environment-variables">Configure the Environment</a>
4. <a href="#add-usb-rules">Add USB rules for an Intel® Neural Compute Stick 2 device (Optional)</a>
6. <a href="#workflow-for-raspberry-pi">Learn About Workflow for Raspberry Pi (Optional)</a>

## <a name="install-openvino"></a>Step 1: Install the Intel® Distribution of OpenVINO™ Toolkit

1. Open the Terminal or your preferred console application.
2. Go to the directory in which you downloaded the Intel® Distribution of OpenVINO™ toolkit. This document assumes this is your `~/Downloads` directory. If not, replace `~/Downloads` with the directory where the file is located.
   ```sh
   cd ~/Downloads/
   ```
   By default, the package file is saved as `l_openvino_toolkit_runtime_raspbian_p_<version>.tgz`.
3. Create an installation folder.
   ```sh
   sudo mkdir -p /opt/intel/openvino_2022
   ```
4. Unpack the archive:
   ```sh
   sudo tar -xf l_openvino_toolkit_runtime_raspbian_p_<version>.tgz --strip 2 -C /opt/intel/openvino_2022
   ```

Now the OpenVINO™ toolkit components are installed. Additional configuration steps are still required. Continue to the next sections to install External Software Dependencies, configure the environment and set up USB rules.

## <a name="install-external-dependencies"></a>Step 2: Install External Software Dependencies

CMake version 3.7.2 or higher is required for building the OpenVINO™ toolkit sample application. To install, open a Terminal window and run the following command:
```sh
sudo apt install cmake
```

CMake is installed. Continue to the next section to set the environment variables.

## <a name="set-the-environment-variables"></a>Step 3: Set the Environment Variables

You must update several environment variables before you can compile and run OpenVINO™ toolkit applications. Run the following script to temporarily set the environment variables:
```sh
source /opt/intel/openvino_2022/setupvars.sh
```

If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source /opt/intel/openvino_2022/bin/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO™ versions on your machine and want to switch among them, as each may require different setup.

The environment variables are set. Next, you can download some additional tools.

## <a name="add-usb-rules"></a>Step 4 (Optional): Add USB Rules for an Intel® Neural Compute Stick 2 device

@sphinxdirective

Only if you want to perform inference on Intel® Neural Compute Stick 2, follow the steps on :ref:`NCS2 Setup Guide <ncs guide raspbianos>`.

@endsphinxdirective

## <a name="workflow-for-raspberry-pi"></a>Step 5 (Optional): Workflow for Raspberry Pi

If you want to use your model for inference, the model must be converted to the .bin and .xml Intermediate Representation (IR) files that are used as input by OpenVINO Runtime. The installation on Raspberry Pi only includes OpenVINO Runtime. Model Optimizer is available on [pypi.org](https://pypi.org/project/openvino-dev/). To get the optimized models, you can use one of the following options:

* Download public and Intel's pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using [Model Downloader tool](@ref omz_tools_downloader). For more information on pre-trained models, see [Pre-Trained Models Documentation](@ref omz_models_group_intel)

* Convert the models using the Model Optimizer.
