# Install OpenVINO™ Runtime on Linux from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_linux}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime. The archive files contain pre-built binaries and library files needed for OpenVINO Runtime, as well as sample code for running demos. This page provides instructions showing how to install OpenVINO Runtime using archive files. Check the Release Notes for more information on updates in the 2022.2 release.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Ubuntu 18.04 long-term support (LTS), 64-bit
  * Ubuntu 20.04 long-term support (LTS), 64-bit
  * Red Hat Enterprise Linux 8, 64-bit
  * Debian 9 armhf

  .. note::
     Since the OpenVINO™ 2022.1 release, CentOS 7.6, 64-bit is not longer supported.

.. tab:: Hardware

  Optimized for these processors:

  * 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
  * 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
  * Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
  * Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
  * Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
  * Intel® Iris® Xe MAX Graphics
  * Intel® Neural Compute Stick 2
  * Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

.. tab:: Processor Notes

  Processor graphics are not included in all processors. 
  See `Product Specifications`_ for information about your processor.
  
  .. _Product Specifications: https://ark.intel.com/

.. tab:: Software

  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.6 - 3.9, 64-bit <https://www.python.org/downloads/windows/>`_
     * Note that OpenVINO is gradually phasing out support for Python 3.6. Python 3.7 - 3.9 are recommended. 

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-openvino"></a>Step 1: Download and Install the OpenVINO Core Components

First, open a terminal using Ctrl + Alt + T. Create a folder for OpenVINO and move into it by issuing the following commands. If the `/opt/intel` folder already exists, skip the `mkdir` command.

```sh
sudo mkdir /opt/intel
cd /opt/intel
```

> **NOTE**: The `/opt/intel` path is the recommended folder path for administrators or root users. If you prefer to install OpenVINO in regular userspace, the recommended path is `/home/<USER>/Intel`. You may use a different path if desired.

Next, you'll download the OpenVINO Runtime 2022.2 archive file from the [OpenVINO archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/windows) site. Issue the following command to download the archive file, unpack it, and rename the folder to `openvino_2022.2.0.7713`:

@sphinxdirective

.. tab:: Ubuntu 18.04

   .. code-block:: sh
   
      sudo wget https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/l_openvino_toolkit_ubuntu18_2022.2.0.7713.af16ea1d79a_x86_64.tgz -O openvino_2022.2.0.7713.tgz
      sudo tar -xf openvino_2022.2.0.7713.tgz
      sudo mv l_openvino_toolkit_ubuntu18_2022.2.0.7713.af16ea1d79a_x86_64 openvino_2022.2.0.7713
      
.. tab:: Ubuntu 20.04

   .. code-block:: sh
   
      sudo wget https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/l_openvino_toolkit_ubuntu20_2022.2.0.7713.af16ea1d79a_x86_64.tgz -O openvino_2022.2.0.7713.tgz
      sudo tar -xf openvino_2022.2.0.7713.tgz
      sudo mv l_openvino_toolkit_ubuntu20_2022.2.0.7713.af16ea1d79a_x86_64 openvino_2022.2.0.7713
      
.. tab:: Red Hat

   .. code-block:: sh
   
      sudo wget https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/l_openvino_toolkit_rhel8_2022.2.0.7713.af16ea1d79a_x86_64.tgz -O openvino_2022.2.0.7713.tgz
      sudo tar -xf openvino_2022.2.0.7713.tgz
      sudo mv l_openvino_toolkit_rhel8_2022.2.0.7713.af16ea1d79a_x86_64 openvino_2022.2.0.7713
      
.. tab:: Debian

   .. code-block:: sh
   
      sudo wget https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/l_openvino_toolkit_debian9_arm_2022.2.0.7713.af16ea1d79a_armhf.tgz -O openvino_2022.2.0.7713.tgz
      sudo tar -xf openvino_2022.2.0.7713.tgz
      sudo mv l_openvino_toolkit_debian9_arm_2022.2.0.7713.af16ea1d79a_armhf openvino_2022.2.0.7713
     
@endsphinxdirective

Finally, create a symbolic link to the folder by issuing:

```
sudo ln -s openvino_2022.2.0.7713 openvino_2022
```
> **NOTE**: If you have already installed a previous release of OpenVINO 2022, a symbolic link to the `openvino_2022` folder may already exist. Remove the previous link with `sudo rm openvino_2022`, then re-issue the previous command.

Congratulations, you finished installation! The `/opt/intel/openvino_2022` folder now contains the core components for OpenVINO™. When other pages in OpenVINO™ documentation refer to the <INSTALL_DIR> directory, this is the folder they're referring to. If you installed OpenVINO™ in a different location (such as `/home/<USER>/Intel/openvino_2022`), make sure to use that instead.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Open a terminal (if it isn't already open) and run the setupvars.bat batch file as shown below to temporarily set your environment variables. Again, if you installed OpenVINO™ in a folder other than `/opt/intel/openvino_2022`, use that location instead.

```sh
source /opt/intel/openvino_2022/setupvars.sh
```  

If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source /opt/intel/openvino_2022/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO™ versions on your machine and want to switch among them, as each may require different setup.

The environment variables are set. Next, you can download some additional tools.

### <a name="model-optimizer">Step 3 (Optional): Install Additional Components
OpenVINO Development Tools adds even more functionality to OpenVINO. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately. 

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on Github](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

### <a name="optional-steps"></a>Step 4 (Optional): Configure Inference on Non-CPU Devices

@sphinxdirective 
.. tab:: GPU

   To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide>`.

.. tab:: NCS 2

   To perform inference on Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X VPU, follow the steps on :ref:`NCS2 Setup Guide <ncs guide>`.
   <!--For more details, see the `Get Started page for Intel® Neural Compute Stick 2 <https://software.intel.com/en-us/neural-compute-stick/get-started>`.-->

.. tab:: VPU

   To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the :ref:`VPU Configuration Guide <vpu guide>`.
   After configuration is done, you are ready to run the verification scripts with the HDDL Plugin for your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 

   .. warning::
      While working with either HDDL or NCS, choose one of them as they cannot run simultaneously on the same machine.

.. tab:: GNA

   To enable the toolkit components to use Intel® Gaussian & Neural Accelerator (GNA) on your system, follow the steps in :ref:`GNA Setup Guide <gna guide>`.
   
@endsphinxdirective

## <a name="get-started"></a>What's Next?

Now you are ready to try out the toolkit.

Start with some Python tutorials:
   * [Hello Image Classification](https://docs.openvino.ai/latest/notebooks/001-hello-world-with-output.html)
   * [Convert TensorFlow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Convert a PyTorch model and remove the image background](https://docs.openvino.ai/latest/notebooks/205-vision-background-removal-with-output.html)

To start with C++ samples, see <a href="openvino_docs_OV_UG_Samples_Overview.html#build-samples-linux">Build Sample Applications on Linux</a> first, and then you can try the following samples:
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)

## <a name="uninstall"></a>Uninstalling the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## Additional Resources

@sphinxdirective
      
* :ref:`Troubleshooting Guide for OpenVINO Installation & Configuration <troubleshooting guide for install>`
* Converting models for use with OpenVINO™: :ref:`Model Optimizer User Guide <deep learning model optimizer>`
* Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning openvino runtime>`
* Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
* Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
* IoT libraries and code samples in the GitHUB repository: `Intel® IoT Developer Kit`_ 

.. _Intel® IoT Developer Kit: https://github.com/intel-iot-devkit

@endsphinxdirective
