# Install OpenVINO™ Runtime on Linux from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_linux}


@sphinxdirective

Installing OpenVINO Runtime from archive files is recommended for C++ developers. It will contain code samples, 
as well as pre-built binaries and library files needed for OpenVINO Runtime. If you work with Python, 
the PyPI package may be a better choice. See the :doc:`Install OpenVINO from PyPI <openvino_docs_install_guides_installing_openvino_pip>` 
page for instructions on how to install OpenVINO Runtime for Python using PyPI.

.. note::

   The following development tools can be installed via `pypi.org <https://pypi.org/project/openvino-dev/>`__ only: 
   model conversion API, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, 
   Accuracy Checker, and Annotation Converter.

See the `Release Notes <https://software.intel.com/en-us/articles/OpenVINO-RelNotes>`__ for more information on updates in the latest release.

.. tab-set::

   .. tab-item:: System Requirements
      :sync: sysreq
   
      | Full requirement listing is available in:
      | `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`__
   
   .. tab-item:: Processor Notes
      :sync: processor_notes
   
      | Processor graphics are not included in all processors. 
      | See `Product Specifications <https://ark.intel.com/>`__ for information about your processor.
   
   .. tab-item:: Software
      :sync: soft
   
      * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
      * `Python 3.7 - 3.11, 64-bit <https://www.python.org/downloads/>`__
      * GCC:
      
      .. tab-set::

         .. tab-item:: Ubuntu 18.04
            :sync: ubuntu18
       
            * GCC 7.5.0
       
         .. tab-item:: Ubuntu 20.04
            :sync: ubuntu20
       
            * GCC 9.3.0
       
         .. tab-item:: RHEL 8
            :sync: rhel8
       
            * GCC 8.4.1
       
         .. tab-item:: CentOS 7
            :sync: centos7
       
            * GCC 8.3.1
            Use the following instructions to install it:
        
            Install GCC 8.3.1 via devtoolset-8
        
              .. code-block:: sh
           
                sudo yum update -y && sudo yum install -y centos-release-scl epel-release
                sudo yum install -y devtoolset-8
        
            Enable devtoolset-8 and check current gcc version
        
               .. code-block:: sh
           
                 source /opt/rh/devtoolset-8/enable
                 gcc -v
        
       

Installing OpenVINO Runtime
############################################################

Step 1: Download and Install the OpenVINO Core Components
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T

2. Create the ``/opt/intel`` folder for OpenVINO by using the following command. If the folder already exists, skip this step.

   .. code-block:: sh
   
      sudo mkdir /opt/intel
   
   .. note::
      The ``/opt/intel`` path is the recommended folder path for administrators or root users. If you prefer to install OpenVINO in regular userspace, the recommended path is ``/home/<USER>/intel``. You may use a different path if desired.

3. Browse to the current user's ``Downloads`` folder:
   
   .. code-block:: sh
   
      cd <user_home>/Downloads
    
4. Download the `OpenVINO Runtime archive file for your system <https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/>`_, extract the files, rename the extracted folder and move it to the desired path:

   .. tab-set::

      .. tab-item:: x86_64
         :sync: x86_64

         .. tab-set::

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu22
         
               .. code-block:: sh
            
                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz
                  tar -xf openvino_2023.0.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0
         
            .. tab-item:: Ubuntu 20.04
               :sync: ubuntu20
         
               .. code-block:: sh
            
                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz
                  tar -xf openvino_2023.0.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0
         
            .. tab-item:: Ubuntu 18.04
               :sync: ubuntu18
         
               .. code-block:: sh
            
                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu18_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz
                  tar -xf openvino_2023.0.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu18_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0
         
            .. tab-item:: RHEL 8
               :sync: rhel8
         
               .. code-block:: sh
            
                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_rhel8_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz
                  tar -xf openvino_2023.0.0.tgz
                  sudo mv l_openvino_toolkit_rhel8_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0
         
            .. tab-item:: CentOS 7
               :sync: centos7
         
               .. code-block:: sh
            
                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_centos7_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_2023.0.0.tgz
                  tar -xf openvino_2023.0.0.tgz
                  sudo mv l_openvino_toolkit_centos7_2023.0.0.10926.b4452d56304_x86_64 /opt/intel/openvino_2023.0.0
       
      .. tab-item:: ARM 64-bit
         :sync: arm_64
      
         .. code-block:: sh
      
            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_debian9_2023.0.0.10926.b4452d56304_arm64.tgz -O openvino_2023.0.0.tgz
            tar -xf openvino_2023.0.0.tgz
            sudo mv l_openvino_toolkit_debian9_2023.0.0.10926.b4452d56304_arm64 /opt/intel/openvino_2023.0.0
      
      .. tab-item:: ARM 32-bit
         :sync: arm_32
      
         .. code-block:: sh
      
            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_debian9_2023.0.0.10926.b4452d56304_armhf.tgz -O openvino_2023.0.0.tgz
            tar -xf openvino_2023.0.0.tgz
            sudo mv l_openvino_toolkit_debian9_2023.0.0.10926.b4452d56304_armhf /opt/intel/openvino_2023.0.0
      
      
5. Install required system dependencies on Linux. To do this, OpenVINO provides a script in the extracted installation directory. Run the following command:
   
   .. code-block:: sh

      cd /opt/intel/openvino_2023.0.0
      sudo -E ./install_dependencies/install_openvino_dependencies.sh

6. For simplicity, it is useful to create a symbolic link as below:
   
   .. code-block:: sh
   
      cd /opt/intel
      sudo ln -s openvino_2023.0.0 openvino_2023
  
   .. note::
      If you have already installed a previous release of OpenVINO 2023, a symbolic link to the ``openvino_2023`` folder may already exist. 
      Unlink the previous link with ``sudo unlink openvino_2023``, and then re-run the command above.


Congratulations, you have finished the installation! The ``/opt/intel/openvino_2023`` folder now contains 
the core components for OpenVINO. If you used a different path in Step 2, for example, ``/home/<USER>/intel/``, 
OpenVINO is now in ``/home/<USER>/intel/openvino_2023``. The path to the ``openvino_2023`` directory is 
also referred as ``<INSTALL_DIR>`` throughout the OpenVINO documentation.


Step 2: Configure the Environment
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You must update several environment variables before you can compile and run OpenVINO applications. 
Open a terminal window and run the ``setupvars.sh`` script as shown below to temporarily set your environment variables. 
If your <INSTALL_DIR> is not ``/opt/intel/openvino_2023``, use the correct one instead.

.. code-block:: sh

   source /opt/intel/openvino_2023/setupvars.sh


If you have more than one OpenVINO version installed on your system, you can easily switch versions by sourcing the `setupvars.sh` of your choice.

.. note:: 
   
   The above command must be re-run every time you start a new terminal session. 
   To set up Linux to automatically run the command every time a new terminal is opened, 
   open ``~/.bashrc`` in your favorite editor and add ``source /opt/intel/openvino_2023/setupvars.sh`` after the last line. 
   Next time when you open a terminal, you will see ``[setupvars.sh] OpenVINO™ environment initialized``. 
   Changing ``.bashrc`` is not recommended when you have multiple OpenVINO versions on your machine and want to switch among them.

The environment variables are set. Continue to the next section if you want to download any additional components.

Step 3 (Optional): Install Additional Components
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. 
It provides tools like model conversion API, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. 
If you install OpenVINO Runtime using archive files, OpenVINO Development Tools must be installed separately.

See the :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` 
page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their 
capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the 
`instructions on GitHub <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__.

Step 4 (Optional): Configure Inference on Non-CPU Devices
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

OpenVINO Runtime has a plugin architecture that enables you to run inference on multiple devices 
without rewriting your code. Supported devices include integrated GPUs, discrete GPUs and GNAs. 
See the instructions below to set up OpenVINO on these devices.

.. tab-set::

   .. tab-item:: GPU
      :sync: gpu
   
      To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide>`.
   
   .. tab-item:: GNA
      :sync: gna
   
      To enable the toolkit components to use Intel® Gaussian & Neural Accelerator (GNA) on your system, follow the steps in :ref:`GNA Setup Guide <gna guide>`.
      


What's Next?
############################################################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! 
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

.. tab-set::

   .. tab-item:: Get started with Python
      :sync: get_started_python
      
      Try the `Python Quick Start Example <notebooks/201-vision-monodepth-with-output.html>`_
      to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.
      
      .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
         :width: 400
      
      Visit the :doc:`Tutorials <tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:
      
      * `OpenVINO Python API Tutorial <notebooks/002-openvino-api-with-output.html>`__
      * `Basic image classification program with Hello Image Classification <notebooks/001-hello-world-with-output.html>`__
      * `Convert a PyTorch model and use it for image background removal <notebooks/205-vision-background-removal-with-output.html>`__
   
   
   .. tab-item:: Get started with C++
      :sync: get_started_cpp
   
      Try the :doc:`C++ Quick Start Example <openvino_docs_get_started_get_started_demos>` for step-by-step instructions 
      on building and running a basic image classification C++ application.
      
      .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
         :width: 400
   
      Visit the :doc:`Samples <openvino_docs_OV_UG_Samples_Overview>` page for other C++ example applications to get you started with OpenVINO, such as:
      
      * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`__
      * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`__
   
   
   
Uninstalling the Intel® Distribution of OpenVINO™ Toolkit
###########################################################

To uninstall the toolkit, follow the steps on the :doc:`Uninstalling page <openvino_docs_install_guides_uninstalling_openvino>`.


Additional Resources
###########################################################

* :doc:`Troubleshooting Guide for OpenVINO Installation & Configuration <openvino_docs_get_started_guide_troubleshooting>`
* Converting models for use with OpenVINO™: :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
* Writing your own OpenVINO™ applications: :doc:`OpenVINO™ Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`
* Sample applications: :doc:`OpenVINO™ Toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`
* Pre-trained deep learning models: :doc:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model_zoo>`
* IoT libraries and code samples in the GitHub repository: `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__
* `OpenVINO Installation Selector Tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__



@endsphinxdirective

