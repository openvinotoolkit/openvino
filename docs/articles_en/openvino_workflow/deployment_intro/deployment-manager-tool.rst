.. {#openvino_docs_install_guides_deployment_manager_tool}

Deploying Your Application with Deployment Manager
==================================================


.. meta::
   :description: OpenVINO™ Deployment Manager assembles the model, OpenVINO IR 
                 files, your application, dependencies and creates a deployment 
                 package for a target device.


.. warning::

   Note that OpenVINO Deployment Manager is deprecated and will be removed in OpenVINO 2024.0.

The OpenVINO™ Deployment Manager is a Python command-line tool that creates a deployment package by assembling the model, OpenVINO IR files, your application, and associated dependencies into a runtime package for your target device. This tool is delivered within the Intel® Distribution of OpenVINO™ toolkit for Linux, Windows and macOS release packages. It is available in the ``<INSTALL_DIR>/tools/deployment_manager`` directory after installation.

This article provides instructions on how to create a package with Deployment Manager and then deploy the package to your target systems.

Prerequisites
####################

To use the Deployment Manager tool, the following requirements need to be met:

* Intel® Distribution of OpenVINO™ toolkit is installed. See the :doc:`Installation Guide <openvino_docs_install_guides_overview>` for instructions on different operating systems.
* To run inference on a target device other than CPU, device drivers must be pre-installed:

  * **For GPU**, see :doc:`Configurations for Intel® Processor Graphics (GPU) <openvino_docs_install_guides_configurations_for_intel_gpu>`.
  * **For GNA**, see :doc:`Intel® Gaussian & Neural Accelerator (GNA) <openvino_docs_install_guides_configurations_for_intel_gna>`

.. important::

   The operating system on the target system must be the same as the development system on which you are creating the package. For example, if the target system is Ubuntu 18.04, the deployment package must be created from the OpenVINO™ toolkit installed on Ubuntu 18.04.

.. tip::

   If your application requires additional dependencies, including the Microsoft Visual C++ Redistributable, use the `'--user_data' option <#running-deployment-manager-in-standard-cli-mode>`__ to add them to the deployment archive. Install these dependencies on the target host before running inference.

Creating Deployment Package Using Deployment Manager
####################################################

To create a deployment package that includes inference-related components of OpenVINO™ toolkit, you can run the Deployment Manager tool in either interactive or standard CLI mode .

Running Deployment Manager in Interactive Mode
++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: Click to expand/collapse

   The interactive mode provides a user-friendly command-line interface that guides through the process with text prompts.

   To launch the Deployment Manager in interactive mode, open a new terminal window, go to the Deployment Manager tool directory, and run the tool script without parameters:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows
   
         .. code-block:: bat
   
            cd <INSTALL_DIR>\tools\deployment_manager
            .\deployment_manager.py

      .. tab-item:: Linux
         :sync: linux
   
         .. code-block:: sh
   
            cd <INSTALL_DIR>/tools/deployment_manager
            ./deployment_manager.py
   
      .. tab-item:: macOS
         :sync: macos
   
         .. code-block:: sh
   
            cd <INSTALL_DIR>/tools/deployment_manager
            ./deployment_manager.py


   The target device selection dialog is displayed:

   .. image:: _static/images/selection_dialog.png
      :alt: Deployment Manager selection dialog

   Use the options provided on the screen to complete the selection of the target devices, and press **Enter** to proceed to the package generation dialog. To interrupt the generation    process and exit the program, type **q** and press **Enter**.

   Once the selection is accepted, the package generation dialog will appear:

   .. image:: _static/images/configuration_dialog.png
      :alt: Deployment Manager configuration dialog

   The target devices selected in the previous step appear on the screen. To go back and change the selection, type **b** and press **Enter**. Use the default settings, or use the    following options to configure the generation process:

   * ``o. Change output directory`` (optional): the path to the output directory. By default, it is set to your home directory.

   * ``u. Provide (or change) path to folder with user data`` (optional): the path to a directory with user data (OpenVINO IR, model, dataset, etc.) files and subdirectories required    for inference, which will be added to the deployment archive. By default, it is set to ``None``, which means that copying the user data to the target system need to be done    separately.

   * ``t. Change archive name`` (optional): the deployment archive name without extension. By default, it is set to ``openvino_deployment_package``.

   After all the parameters are set, type **g** and press **Enter** to generate the package for the selected target devices. To interrupt the generation process and exit the program,    type **q** and press **Enter**.

   Once the script has successfully completed, the deployment package is generated in the specified output directory.


Running Deployment Manager in Standard CLI Mode
+++++++++++++++++++++++++++++++++++++++++++++++


.. dropdown:: Click to expand/collapse

   You can also run the Deployment Manager tool in the standard CLI mode. In this mode, specify the target devices and other parameters as command-line arguments of the Deployment Manager Python script. This mode facilitates integrating the tool in an automation pipeline.

   To launch the Deployment Manager tool in the standard mode: open a new terminal window, go to the Deployment Manager tool directory, and run the tool command with the following    syntax:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows
   
         .. code-block:: bat
   
            cd <INSTALL_DIR>\tools\deployment_manager
            .\deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]

      .. tab-item:: Linux
         :sync: linux
   
         .. code-block:: sh
   
            cd <INSTALL_DIR>/tools/deployment_manager
            ./deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]
   
      .. tab-item:: macOS
         :sync: macos
   
         .. code-block:: sh
   
            cd <INSTALL_DIR>/tools/deployment_manager
            ./deployment_manager.py <--targets> [--output_dir] [--archive_name] [--user_data]


   The following options are available:

   * ``<--targets>`` (required): the list of target devices to run inference. To specify more than one target, separate them with spaces, for example, ``--targets cpu gpu``.
   To get a list of currently available targets, run the program with the ``-h`` option.

   * ``[--output_dir]`` (optional): the path to the output directory. By default, it is set to your home directory.

   * ``[--archive_name]`` (optional): a deployment archive name without extension. By default, it is set to ``openvino_deployment_package``.

   * ``[--user_data]`` (optional): the path to a directory with user data (OpenVINO IR, model, dataset, etc.) files and subdirectories required for inference, which will be added to the    deployment archive. By default, it is set to ``None``, which means copying the user data to the target system need to be performed separately.

   Once the script has successfully completed, the deployment package is generated in the output directory specified.


Deploying Package on Target Systems
###################################

Once the Deployment Manager has successfully completed, the ``.tar.gz`` (on Linux or macOS) or ``.zip`` (on Windows) package is generated in the specified output directory.

To deploy the OpenVINO Runtime components from the development machine to the target system, perform the following steps:

1. Copy the generated archive to the target system by using your preferred method.

2. Extract the archive to the destination directory on the target system. If the name of your archive is different from the default one shown below, replace ``openvino_deployment_package`` with your specified name.

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows
   
         .. code-block:: bat
   
            Use the archiver of your choice to unzip the file.

      .. tab-item:: Linux
         :sync: linux
   
         .. code-block:: sh
   
            tar xf openvino_deployment_package.tar.gz -C <destination_dir>
   
      .. tab-item:: macOS
         :sync: macos
   
         .. code-block:: sh
   
            tar xf openvino_deployment_package.tar.gz -C <destination_dir>


   Now, the package is extracted to the destination directory. The following files and subdirectories are created:

   * ``setupvars.sh`` — a copy of ``setupvars.sh``.
   * ``runtime`` — contains the OpenVINO runtime binary files.
   * ``install_dependencies`` — a snapshot of the ``install_dependencies`` directory from the OpenVINO installation directory.
   * ``<user_data>`` — the directory with the user data (OpenVINO IR, model, dataset, etc.) specified while configuring the package.

3. On a target Linux system, to run inference install additional dependencies by running the ``install_openvino_dependencies.sh`` script:

   .. code-block:: sh

      cd <destination_dir>/openvino/install_dependencies
      sudo -E ./install_openvino_dependencies.sh


4. Set up the environment variables:

   .. tab-set::
   
      .. tab-item:: Windows
         :sync: windows
      
         .. code-block:: bat
      
            cd <destination_dir>\openvino\
            .\setupvars.bat

      .. tab-item:: Linux
         :sync: linux
      
         .. code-block:: sh
      
            cd <destination_dir>/openvino/
            source ./setupvars.sh
      
      .. tab-item:: macOS
         :sync: macos
      
         .. code-block:: sh
      
            cd <destination_dir>/openvino/
            source ./setupvars.sh


Now, you have finished the deployment of the OpenVINO Runtime components to the target system.

