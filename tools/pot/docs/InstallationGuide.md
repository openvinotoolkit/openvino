# Post-Training Optimization Tool Installation Guide {#pot_InstallationGuide}

## Prerequisites

* Python* 3.6 or higher
* [OpenVINO&trade;](https://docs.openvino.ai/latest/index.html)

The minimum and the recommended requirements to run the Post-training Optimization Tool (POT) are the same as in [OpenVINO&trade;](https://docs.openvino.ai/latest/index.html).

There are two ways how to install the POT on your system:
- Installation from PyPI repository
- Installation from Intel&reg; Distribution of OpenVINO&trade; toolkit package

## Install POT from PyPI
The simplest way to get the Post-training Optimization Tool and OpenVINO&trade; installed is to use PyPI. Follow the steps below to do that:
1. Create a separate [Python* environment](https://docs.python.org/3/tutorial/venv.html) and activate it
2. To install OpenVINO&trade;, run `pip install openvino`.
3. To install POT and other OpenVINO&trade; developer tools, run `pip install openvino-dev`.

Now the Post-training Optimization Tool is available in the command line by the `pot` alias. To verify it, run `pot -h`.

## Install and Set Up POT from Intel&reg; Distribution of OpenVINO&trade; toolkit package

In the instructions below, `<INSTALL_DIR>` is the directory where the Intel&reg; distribution of OpenVINO&trade; toolkit
is installed. The Post-training Optimization Tool is distributed as a part of the OpenVINO&trade; release package, and to use the POT as a command-line tool,
you need to install OpenVINO&trade; as well as POT dependencies, namely [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide)
and [Accuracy Checker](@ref omz_tools_accuracy_checker). It is recommended to create a separate [Python* environment](https://docs.python.org/3/tutorial/venv.html) before installing the OpenVINO&trade; and its components.
POT source files are available in `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` directory after the OpenVINO&trade; installation.

To set up the Post-training Optimization Tool in your environment, follow the steps below.

### Set up the Model Optimizer and Accuracy Checker components

- To set up the [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide):
   1. Go to `<INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites`.
   2. Run the following script to configure the Model Optimizer:
      * Linux: 
      ```sh 
      sudo ./install_prerequisites.sh
      ```  
      * Windows: 
      ```bat 
      install_prerequisites.bat
      ```
   3. To verify that the Model Optimizer is installed, run `<INSTALL_DIR>/deployment_tools/model_optimizer/mo.py -h`.
  
- To set up the [Accuracy Checker](@ref omz_tools_accuracy_checker):
   1. Go to `<INSTALL_DIR>/deployment_tools/open_model_zoo/tools/accuracy_checker`.
   2. Run the following script to configure the Accuracy Checker:
   ```sh
   python setup.py install
   ```
   3. Now the Accuracy Checker is available in the command line by the `accuracy_check` alias. To verify it, run `accuracy_check -h`.

### Set up the POT

1. Go to `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit`.
2. Run the following script to configure the POT:
   ```sh
   python setup.py install
   ```

   In order to enable advanced algorithms such as the Tree-Structured Parzen Estimator (TPE) based optimization, add the following flag to the installation command:
   ```sh
   python setup.py install --install-extras
   ```
3. Now the POT is available in the command line by the `pot` alias. To verify it, run `pot -h`.
