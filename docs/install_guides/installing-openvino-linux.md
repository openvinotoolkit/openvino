# Install and Configure Intel® Distribution of OpenVINO™ Toolkit for Linux {#openvino_docs_install_guides_installing_openvino_linux}

> **NOTE**: With the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

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

@endsphinxdirective

## Installation Flow

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#install-external-dependencies">Install External Software Dependencies</a>
3. <a href="#set-the-environment-variables">Configure the Environment</a>
4. <a href="#model-optimizer">Download additional components (Optional)</a>
5. <a href="#optional-steps">Configure Inference on non-CPU Devices (Optional)</a>
6. <a href="#get-started">What's next?</a>

@sphinxdirective

.. important::
   Before installation of the Intel® Distribution of OpenVINO™, you may check our prepared :ref:`code samples <code samples>` in C, C++, Python and :ref:`notebook tutorials <notebook tutorials>`, so you can check capabilities of OpenVINO™ toolkit.

@endsphinxdirective

## <a name="install-openvino"></a>Step 1: Install the Intel® Distribution of OpenVINO™ Toolkit

1. Download the Intel® Distribution of OpenVINO™ toolkit installer file from [Intel® Distribution of OpenVINO™ toolkit for Linux](https://software.intel.com/en-us/openvino-toolkit/choose-download).
2. Open a Linux command line window with keyboard shortcut: *Ctrl+Alt+T*
3. Change directory position to the one in which you saved toolkit file.<br>
   ```sh
   cd ~/<SAVE_DIR>
   ```
   **EXAMPLE:** If you downloaded the starter script to the current user `Downloads` directory:
   ```sh
   cd ~/Downloads/
   ```
   There will be a bootstrapper script `l_openvino_toolkit_p_<version>.sh`.
4. Add executable rights for the current user:
   ```sh
   chmod +x l_openvino_toolkit_p_<version>.sh
   ```
5. You can run installation now in:
   + Command Line (CLI), by adding parameters `-a` for additional arguments and `--cli` to run installation in command line:
     ```sh
     ./l_openvino_toolkit_p_<version>.sh -a --cli
     ```
     > **NOTE**: To get additional information on all parameters that can be used, use the help option: `--help`.

   + Graphical User Interface (GUI), by running script without any additional parameters:
     ```sh
     ./l_openvino_toolkit_p_<version>.sh
     ```
     <br>Installation dialog box will start up:

      @sphinxdirective

      .. image:: _static/images/openvino-install.png
         :width: 400px
         :align: center

      @endsphinxdirective
   
6. Follow the instructions on your screen. During the installation you will be asked to accept the license agreement. Your acceptance is required to continue. Check the installation process on the image below:<br>

   ![](../img/openvino-install-linux-run-boostrapper-script.gif)
   Click on the image to see the details.
   <br>
   <br>By default, the Intel® Distribution of OpenVINO™ is installed to the following directory, referred to as `<INSTALL_DIR>` elsewhere in the documentation:
   * For root or administrator: `/opt/intel/openvino_<version>/`
   * For regular users: `/home/<USER>/intel/openvino_<version>/`

   <br>For simplicity, a symbolic link to the latest installation is also created: `/opt/intel/openvino_2022/` or `/home/<USER>/intel/openvino_2022/`.

To check **Release Notes** please visit: [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes).

The core components are now installed. Continue to the next section to install additional dependencies.

## <a name="install-external-dependencies"></a>Step 2: Install External Software Dependencies

1. Go to the `install_dependencies` directory:
   ```sh
   cd <INSTALL_DIR>/install_dependencies
   ```
2. Run a script to download and install the external software dependencies:
   ```sh
   sudo -E ./install_openvino_dependencies.sh
   ```

## <a name="set-the-environment-variables"></a>Step 3: Configure the Environment

To compile and run OpenVINO™ applications, update environment variables:

```sh
source <INSTALL_DIR>/setupvars.sh
```  

@sphinxdirective

.. important::
   In case you have more than one OpenVINO™ version on your machine, you can switch between them by sourcing `setupvars.sh` of your choice.

@endsphinxdirective

<!-- > **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source <INSTALL_DIR>/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO™ versions on your machine and want to switch among them, as each may require different setup. -->

## <a name="model-optimizer">Step 4 (Optional): Download Additional Components

@sphinxdirective

.. dropdown:: OpenCV

   OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. The Intel® Distribution of OpenVINO™ provides a script to install OpenCV: ``<INSTALL_DIR>/extras/scripts/download_opencv.sh``.

   .. note::
      Make sure you have 2 prerequisites installed: ``curl`` and ``tar``.

   Depending on how you have installed the Intel® Distribution of OpenVINO™, the script should be run either as root or regular user. After the execution of the script, you will find OpenCV extracted to ``<INSTALL_DIR>/extras/opencv``.

@endsphinxdirective

## <a name="optional-steps"></a>Step 5 (Optional): Configure Inference on Non-CPU Devices

@sphinxdirective
.. tab:: GNA

   To enable the toolkit components to use Intel® Gaussian & Neural Accelerator (GNA) on your system, follow the steps in :ref:`GNA Setup Guide <gna guide>`.
   
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

@endsphinxdirective

## <a name="get-started"></a>Step 6: What's Next?

Now you are ready to try out the toolkit.

Developing in Python:
   * [Start with TensorFlow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)

## <a name="uninstall"></a>Uninstalling the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## Additional Resources

@sphinxdirective

.. dropdown:: Troubleshooting

   PRC developers might encounter pip errors during Intel® Distribution of OpenVINO™ installation. To resolve the issues, try one of the following options:
   
   * Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   .. code-block:: sh

      pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/

   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   
   * If you run into incompatibility issues between components after installing new Intel® Distribution of OpenVINO™ version, try running ``requirements.txt`` with the following command:

   .. code-block:: sh

      pip install -r <INSTALL_DIR>/tools/requirements.txt

@endsphinxdirective

@sphinxdirective

.. dropdown:: Additional Resources
      
   * Converting models for use with OpenVINO™: :ref:`Model Optimizer Developer Guide <deep learning model optimizer>`
   * Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning inference engine>`
   * Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
   * Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
   * IoT libraries and code samples in the GitHUB repository: `Intel® IoT Developer Kit`_ 

<!---
   To learn more about converting models from specific frameworks, go to:  
   * :ref:`Convert Your Caffe Model <convert model caffe>`
   * :ref:`Convert Your TensorFlow Model <convert model tf>`
   * :ref:`Convert Your MXNet Modele <convert model mxnet>`
   * :ref:`Convert Your Kaldi Model <convert model kaldi>`
   * :ref:`Convert Your ONNX Model <convert model onnx>`
--->   
   .. _Intel® IoT Developer Kit: https://github.com/intel-iot-devkit

@endsphinxdirective
