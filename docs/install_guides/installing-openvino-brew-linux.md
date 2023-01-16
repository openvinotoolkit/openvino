# Install OpenVINO™ Runtime on Linux via HomeBrew {#openvino_docs_install_guides_installing_openvino_brew_linux}

With the OpenVINO™ 2022.3 release, you can install OpenVINO Runtime on Linux via [Homebrew](https://brew.sh/).

> **NOTE**: Only CPU is supported for inference with OpenVINO that is installed via HomeBrew.

Installing OpenVINO Runtime from HomeBrew is recommended for C++ developers. If you are working with Python, the PyPI package has everything needed for Python development and deployment on CPU and GPUs. See the [Install OpenVINO from PyPI](installing-openvino-pip.md) page for instructions on how to install OpenVINO Runtime for Python using PyPI.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

See the [Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-2022-3-lts-relnotes.html) for more information on updates in the latest release.


@sphinxdirective
.. tab:: System Requirements

   | Full requirement listing is available in:
   | `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`_

.. tab:: Processor Notes

  Processor graphics are not included in all processors. 
  See `Product Specifications`_ for information about your processor.
  
  .. _Product Specifications: https://ark.intel.com/

.. tab:: Software

  * [Homebrew](https://brew.sh/)
  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-openvino"></a>Step 1: Download and Install the OpenVINO Core Components

@sphinxdirective

1. Make sure that you have installed HomeBrew on your system. If not, you can run the following command in a prompt terminal window to install it:

   .. code-block:: sh

      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Run the following command to install OpenVINO Runtime:

   .. code-block:: sh

      brew install openvino

   This will install OpenVINO to this directory: `<placehoder>`. 

3. Install required system dependencies on Linux. To do this, OpenVINO provides a script in the extracted installation directory. Run the following command:
   
   .. code-block:: sh
   
      cd /opt/intel/openvino_2022.3.0/
      sudo -E ./install_dependencies/install_openvino_dependencies.sh 

4. For simplicity, it is useful to create a symbolic link as below:
   
   .. code-block:: sh
   
      sudo ln -s openvino_2022.3.0 openvino_2022
  
   .. note::
      If you have already installed a previous release of OpenVINO 2022, a symbolic link to the ``openvino_2022`` folder may already exist. Unlink the previous link with ``sudo unlink openvino_2022``, and then re-run the command above.

@endsphinxdirective

Congratulations, you finished the installation! The `/opt/intel/openvino_2022` folder now contains the core components for OpenVINO. If you used a different path in Step 2, for example, `/home/<USER>/Intel/`, OpenVINO is then installed in `/home/<USER>/Intel/openvino_2022`. The path to the `openvino_2022` directory is also referred as `<INSTALL_DIR>` throughout the OpenVINO documentation.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO applications. Open a terminal window and run the `setupvars.sh` script as shown below to temporarily set your environment variables. If your <INSTALL_DIR> is not `/opt/intel/openvino_2022`, use the correct one instead.

```sh
source /opt/intel/openvino_2022/setupvars.sh
```  

If you have more than one OpenVINO version on your machine, you can easily switch its version by sourcing the `setupvars.sh` of your choice.

> **NOTE**: The above command must be re-run every time you start a new terminal session. To set up Linux to automatically run the command every time a new terminal is opened, open `~/.bashrc` in your favorite editor and add `source /opt/intel/openvino_2022/setupvars.sh` after the last line. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have multiple OpenVINO versions on your machine and want to switch among them.

The environment variables are set. Continue to the next section if you want to download any additional components.

### <a name="model-optimizer">Step 3 (Optional): Install Additional Components
OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Runtime using archive files, OpenVINO Development Tools must be installed separately.

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on GitHub](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

## <a name="uninstall"></a>Uninstalling OpenVINO

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## Additional Resources

@sphinxdirective
      
* :ref:`Troubleshooting Guide for OpenVINO Installation & Configuration <troubleshooting guide for install>`
* Converting models for use with OpenVINO™: :ref:`Model Optimizer User Guide <deep learning model optimizer>`
* Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning openvino runtime>`
* Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
* Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
* IoT libraries and code samples in the GitHub repository: `Intel® IoT Developer Kit`_ 

.. _Intel® IoT Developer Kit: https://github.com/intel-iot-devkit

@endsphinxdirective
