# OpenVINO™ Deployment Manager Guide {#openvino_docs_install_guides_deployment_manager_tool}

The Deployment Manager of Intel® Distribution of OpenVINO™ creates a deployment package by assembling the model, IR files, your application, and associated dependencies into a runtime package for your target device.

The Deployment Manager is a Python\* command-line tool that is delivered within the Intel® Distribution of OpenVINO™ toolkit for Linux\* and Windows\* release packages and available after installation in the `<INSTALL_DIR>/deployment_tools/tools/deployment_manager` directory.

## Pre-Requisites

* Intel® Distribution of OpenVINO™ toolkit for Linux\* (version 2019 R3 or higher) or Intel® Distribution of OpenVINO™ toolkit for Windows\* (version 2019 R4 or higher) installed on your development machine.
* Python\* 3.6 or higher is required to run the Deployment Manager.
* To run inference on a target device other than CPU, device drivers must be pre-installed:
   * For **Linux**, see the following sections in the [installation instructions for Linux](../install_guides/installing-openvino-linux.md): 
     * Steps for Intel® Processor Graphics (GPU) section 
     * Steps for Intel® Neural Compute Stick 2 section
     * Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
   * **For Windows**, see the following sections in the [installation instructions for Windows](../install_guides/installing-openvino-windows.md):
     * Steps for Intel® Processor Graphics (GPU)
     * Steps for the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
     

> **IMPORTANT**: The operating system on the target host must be the same as the development system on which you are creating the package. For example, if the target system is Ubuntu 18.04, the deployment package must be created from the OpenVINO™ toolkit installed on Ubuntu 18.04.     

## Create Deployment Package Using Deployment Manager

There are two ways to create a deployment package that includes inference-related components of the OpenVINO™ toolkit: <br>
You can run the Deployment Manager tool in either Interactive or Standard CLI mode.

### Run Interactive Mode
<details>
  <summary>Click to expand/collapse</summary>
  
Interactive mode provides a user-friendly command-line interface that will guide you through the process with text prompts.

1. To launch the Deployment Manager in the interactive mode, open a new terminal window, go to the Deployment Manager tool directory and run the tool script without parameters:
   ```sh
   <INSTALL_DIR>/deployment_tools/tools/deployment_manager
   ```
   ```sh
   ./deployment_manager.py
   ``` 
2. The target device selection dialog is displayed:
![Deployment Manager selection dialog](../img/selection_dialog.png)
Use the options provided on the screen to complete selection of the target devices and press **Enter** to proceed to the package generation dialog. if you want to interrupt the generation process and exit the program, type **q** and press **Enter**.
3. Once you accept the selection, the package generation dialog is displayed:
![Deployment Manager configuration dialog](../img/configuration_dialog.png)
   1. The target devices you have selected at the previous step appear on the screen. If you want to change the selection, type **b** and press **Enter** to go back to the previous screen. 
   
   2. Use the options provided to configure the generation process, or use the default settings.
   
   3. Once all the parameters are set, type **g** and press **Enter** to generate the package for the selected target devices. If you want to interrupt the generation process and exit the program, type **q** and press **Enter**.

The script successfully completes and the deployment package is generated in the output directory specified. 
</details>

### Run Standard CLI Mode
<details>
  <summary>Click to expand/collapse</summary>

Alternatively, you can run the Deployment Manager tool in the standard CLI mode. In this mode, you specify the target devices and other parameters as command-line arguments of the Deployment Manager Python script. This mode facilitates integrating the tool in an automation pipeline.

To launch the Deployment Manager tool in the standard mode, open a new terminal window, go to the Deployment Manager tool directory and run the tool command with the following syntax:
```sh
./deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]
```

The following options are available:

* `<--targets>` — (Mandatory) List of target devices to run inference. To specify more than one target, separate them with spaces. For example: `--targets cpu gpu vpu`. You can get a list of currently available targets running the tool's help: 
   ```sh
   ./deployment_manager.py -h
   ```
*	`[--output_dir]` — (Optional) Path to the output directory. By default, it set to your home directory.

*	`[--archive_name]` — (Optional) Deployment archive name without extension. By default, it set to `openvino_deployment_package`.

*	`[--user_data]` — (Optional) Path to a directory with user data (IRs, models, datasets, etc.) required for inference. By default, it's set to `None`, which means that the user data are already present on the target host machine.

The script successfully completes and the deployment package is generated in the output directory specified.
</details>

## Deploy Package on Target Hosts

After the Deployment Manager has successfully completed, you can find the generated `.tar.gz` (for Linux) or `.zip` (for Windows) package in the output directory you specified. 

To deploy the Inference Engine components from the development machine to the target host, perform the following steps:

1. Transfer the generated archive to the target host using your preferred method.

2. Unpack the archive into the destination directory on the target host (if your archive name is different from the default shown below, replace the `openvino_deployment_package` with the name you use).
   * For Linux:
   ```sh
   tar xf openvino_deployment_package.tar.gz -C <destination_dir>
   ```
   * For Windows, use an archiver your prefer.  
   
   The package is unpacked to the destination directory and the following subdirectories are created:
      * `bin` — Snapshot of the `bin` directory from the OpenVINO installation directory.
      * `deployment_tools/inference_engine` — Contains the Inference Engine binary files.
      * `install_dependencies` — Snapshot of the `install_dependencies` directory from the OpenVINO installation directory.
      * `<user_data>` — The directory with the user data (IRs, datasets, etc.) you specified while configuring the package.
3. For Linux, to run inference on a target Intel® GPU, Intel® Movidius™ VPU, or Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, you need to install additional dependencies by running the `install_openvino_dependencies.sh` script:
   ```sh
   cd <destination_dir>/openvino/install_dependencies
   ```
   ```sh
   sudo -E ./install_openvino_dependencies.sh
   ```
4. Set up the environment variables:
   * For Linux:
   ```sh
   cd <destination_dir>/openvino/
   ```
   ```sh
   source ./bin/setupvars.sh
   ```
   * For Windows:
   ```
   cd <destination_dir>\openvino\
   ```
   ```
   .\bin\setupvars.bat
   ```

Congratulations, you have finished the deployment of the Inference Engine components to the target host. 