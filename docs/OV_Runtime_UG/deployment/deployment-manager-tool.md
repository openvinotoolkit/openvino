# Deployment Manager {#openvino_docs_install_guides_deployment_manager_tool}

The Deployment Manager is a Python* command-line tool that creates a deployment package by assembling the model, IR files, your application, and associated dependencies into a runtime package for your target device. This tool is delivered within the Intel® Distribution of OpenVINO™ toolkit for Linux*, Windows* and macOS* release packages and is available after installation in the `<INSTALL_DIR>/tools/deployment_manager` directory.

## Prerequisites

* Intel® Distribution of OpenVINO™ toolkit
* To run inference on a target device other than CPU, device drivers must be pre-installed:
   * **For Linux**, see the following sections in the [installation instructions for Linux](../../install_guides/installing-openvino-linux.md):
     * Steps for [Intel® Processor Graphics (GPU)](../../install_guides/configurations-for-intel-gpu.md) section
     * Steps for [Intel® Neural Compute Stick 2 section](../../install_guides/configurations-for-ncs2.md)
     * Steps for [Intel® Vision Accelerator Design with Intel® Movidius™ VPUs](../../install_guides/installing-openvino-config-ivad-vpu.md)
     * Steps for [Intel® Gaussian & Neural Accelerator (GNA)](../../install_guides/configurations-for-intel-gna.md)
   * **For Windows**, see the following sections in the [installation instructions for Windows](../../install_guides/installing-openvino-windows.md):
     * Steps for [Intel® Processor Graphics (GPU)](../../install_guides/configurations-for-intel-gpu.md)
     * Steps for the [Intel® Vision Accelerator Design with Intel® Movidius™ VPUs](../../install_guides/installing-openvino-config-ivad-vpu.md)
   * **For macOS**, see the following section in the [installation instructions for macOS](../../install_guides/installing-openvino-macos.md):
     * Steps for [Intel® Neural Compute Stick 2 section](../../install_guides/configurations-for-ncs2.md)

> **IMPORTANT**: The operating system on the target system must be the same as the development system on which you are creating the package. For example, if the target system is Ubuntu 18.04, the deployment package must be created from the OpenVINO™ toolkit installed on Ubuntu 18.04.

> **TIP**: If your application requires additional dependencies, including the Microsoft Visual C++ Redistributable, use the ['--user_data' option](https://docs.openvino.ai/latest/openvino_docs_install_guides_deployment_manager_tool.html#run-standard-cli-mode) to add them to the deployment archive. Install these dependencies on the target host before running inference.

## Create Deployment Package Using Deployment Manager

There are two ways to create a deployment package that includes inference-related components of the OpenVINO™ toolkit: you can run the Deployment Manager tool in either interactive or standard CLI mode.

### Run Interactive Mode

@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="Click to expand/collapse">

@endsphinxdirective

Interactive mode provides a user-friendly command-line interface that will guide you through the process with text prompts.

To launch the Deployment Manager in interactive mode, open a new terminal window, go to the Deployment Manager tool directory and run the tool script without parameters:
  
@sphinxdirective
   
.. tab:: Linux  
      
   .. code-block:: sh
      
      cd <INSTALL_DIR>/tools/deployment_manager
         
      ./deployment_manager.py  
         
.. tab:: Windows  
      
   .. code-block:: bat  
         
      cd <INSTALL_DIR>\deployment_tools\tools\deployment_manager
      .\deployment_manager.py  
         
.. tab:: macOS  
      
   .. code-block:: sh
         
      cd <INSTALL_DIR>/tools/deployment_manager
      ./deployment_manager.py  
      
@endsphinxdirective

The target device selection dialog is displayed:
  
![Deployment Manager selection dialog](../img/selection_dialog.png)

Use the options provided on the screen to complete selection of the target devices and press **Enter** to proceed to the package generation dialog. if you want to interrupt the generation process and exit the program, type **q** and press **Enter**.

Once you accept the selection, the package generation dialog is displayed:
  
![Deployment Manager configuration dialog](../img/configuration_dialog.png)

The target devices you have selected at the previous step appear on the screen. To go back and change the selection, type **b** and press **Enter**. Use the options provided to configure the generation process, or use the default settings.
   
* `o. Change output directory` (optional): Path to the output directory. By default, it's set to your home directory.

* `u. Provide (or change) path to folder with user data` (optional): Path to a directory with user data (IRs, models, datasets, etc.) files and subdirectories required for inference, which will be added to the deployment archive. By default, it's set to `None`, which means you will separately copy the user data to the target system.

* `t. Change archive name` (optional): Deployment archive name without extension. By default, it is set to `openvino_deployment_package`.
 
Once all the parameters are set, type **g** and press **Enter** to generate the package for the selected target devices. To interrupt the generation process and exit the program, type **q** and press **Enter**.

The script successfully completes and the deployment package is generated in the specified output directory. 

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective

### Run Standard CLI Mode

@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="Click to expand/collapse">

@endsphinxdirective

Alternatively, you can run the Deployment Manager tool in the standard CLI mode. In this mode, you specify the target devices and other parameters as command-line arguments of the Deployment Manager Python script. This mode facilitates integrating the tool in an automation pipeline.

To launch the Deployment Manager tool in the standard mode, open a new terminal window, go to the Deployment Manager tool directory and run the tool command with the following syntax:

@sphinxdirective

.. tab:: Linux

   .. code-block:: sh

      cd <INSTALL_DIR>/tools/deployment_manager
      ./deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]

.. tab:: Windows

   .. code-block:: bat

      cd <INSTALL_DIR>\tools\deployment_manager
      .\deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]

.. tab:: macOS

   .. code-block:: sh

      cd <INSTALL_DIR>/tools/deployment_manager
      ./deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]

@endsphinxdirective

The following options are available:

* `<--targets>` (required): List of target devices to run inference. To specify more than one target, separate them with spaces. For example: `--targets cpu gpu vpu`. You can get a list of currently available targets by running the program with the `-h` option.

* `[--output_dir]` (optional): Path to the output directory. By default, it is set to your home directory.

* `[--archive_name]` (optional): Deployment archive name without extension. By default, it is set to `openvino_deployment_package`.

* `[--user_data]` (optional): Path to a directory with user data (IRs, models, datasets, etc.) files and subdirectories required for inference, which will be added to the deployment archive. By default, it's set to `None`, which means you will separately copy the user data to the target system.

The script successfully completes, and the deployment package is generated in the output directory specified.

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective

## Deploy Package on Target Systems

After the Deployment Manager has successfully completed, you can find the generated `.tar.gz` (for Linux or macOS) or `.zip` (for Windows) package in the output directory you specified.

To deploy the OpenVINO Runtime components from the development machine to the target system, perform the following steps:

1. Copy the generated archive to the target system using your preferred method.

2. Unpack the archive into the destination directory on the target system (if your archive name is different from the default shown below, replace the `openvino_deployment_package` with the name you use).
@sphinxdirective

.. tab:: Linux

    .. code-block:: sh

        tar xf openvino_deployment_package.tar.gz -C <destination_dir>

.. tab:: Windows

    .. code-block:: bat

        Use the archiver of your choice to unzip the file.

.. tab:: macOS

    .. code-block:: sh

        tar xf openvino_deployment_package.tar.gz -C <destination_dir>

@endsphinxdirective

  The package is unpacked to the destination directory and the following files and subdirectories are created:

   * `setupvars.sh` — Copy of `setupvars.sh`
   * `runtime` — Contains the OpenVINO runtime binary files.
   * `install_dependencies` — Snapshot of the `install_dependencies` directory from the OpenVINO installation directory.
   * `<user_data>` — The directory with the user data (IRs, datasets, etc.) you specified while configuring the package.

For Linux, to run inference on a target Intel® GPU, Intel® Movidius™ VPU, or Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, you need to install additional dependencies by running the `install_openvino_dependencies.sh` script on the target machine:

```sh
cd <destination_dir>/openvino/install_dependencies
sudo -E ./install_openvino_dependencies.sh
```

Set up the environment variables:
  
@sphinxdirective  
      
.. tab:: Linux  
      
   .. code-block:: sh
         
      cd <destination_dir>/openvino/
      source ./setupvars.sh
      
.. tab:: Windows  
      
   .. code-block:: bat  
      
      cd <destination_dir>\openvino\
      .\setupvars.bat
      
.. tab:: macOS  
      
   .. code-block:: sh
         
      cd <destination_dir>/openvino/
      source ./setupvars.sh
      
@endsphinxdirective

You have now finished the deployment of the OpenVINO Runtime components to the target system.
