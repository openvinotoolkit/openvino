# Install OpenVINO™ Runtime on Linux from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_linux}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime. 

You can also check the [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes) for more information on updates in this release.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Ubuntu 18.04 long-term support (LTS), 64-bit
  * Ubuntu 20.04 long-term support (LTS), 64-bit

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
     * Note that OpenVINO is gradually stopping the support for Python 3.6. Python 3.7 - 3.9 are recommended. 

@endsphinxdirective

## Installing OpenVINO Runtime

@sphinxdirective

.. important::
   Before you start your journey with installation of the Intel® Distribution of OpenVINO™ toolkit, we encourage you to check the :ref:`code samples <code samples>` in C, C++, Python and :ref:`notebook tutorials <notebook tutorials>`, so you could see all the amazing things that you can achieve with our tool.

@endsphinxdirective

### <a name="install-openvino"></a>Step 1: Download and Install the OpenVINO Package

1. Select and download the OpenVINO™ archive files from [Intel® Distribution of OpenVINO™ toolkit download page](https://software.intel.com/en-us/openvino-toolkit/choose-download). There are typically two files for you to download: 
   ```sh
   l_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.tgz
   l_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.tgz.sha256
   ``` 
   where the `.sha256` file is used to verify the success of the download process.
2. Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T
3. Change the directory to where you downloaded the archive files.<br>
   For example, if you downloaded the files to the current user's `Downloads` directory, use the following command:
   ```sh
   cd ~/Downloads/
   ```
4. To verify the package by using the `.sha256` file:
   ```sh
   sha256sum -с <archive name>.tgz.sha256
   ```
   If any error message appears, check your network connections, re-download the correct files, and make sure the download process completes successfully.
5. Extract OpenVINO files from the `.tgz` file:
   ```sh
   tar xf <archive name>.tgz -C <destination_dir>
   ```
   where the `<destination_dir>` is the directory that you extract OpenVINO files to. You're recommended to set it as:
   * For root users or administrators: `/opt/intel/`
   * For regular users: `/home/<USER>/intel/`

If you forgot to set the directory in Step 5, you can then use `sudo mv <extracted_folder> /opt/intel` (for root users or administrators), or `mv <extracted_folder> /home/<USER>/intel/` (for regular users) to set that.

For simplicity, it is useful to create a symbolink link:
```sh
ln -s /home/<USER>/intel/<extracted_folder> /home/<USER>/intel/openvino_2022
```
If such link already exists, remove the previous link with `rm /home/<USER>/intel/openvino_2022`.

The `/opt/intel/openvino_<version>/` or `/home/<USER>/intel/openvino_<version>/` will be referred as the standard OpenVINO `<INSTALL_DIR>` in this document.

The core components are now installed. Continue to the next section to install components.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Set environment variables as follows:

```sh
source <INSTALL_DIR>/setupvars.sh
```  

If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source <INSTALL_DIR>/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO™ versions on your machine and want to switch among them, as each may require different setup.

The environment variables are set. Next, you can download some additional tools.

### <a name="model-optimizer">Step 3 (Optional): Install Additional Components

Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can only be installed via PyPI. See [Install OpenVINO™ Development Tools](installing-model-dev-tools.md) for detailed steps.

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
