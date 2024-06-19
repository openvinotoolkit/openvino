.. {#openvino_docs_install_guides_installing_openvino_from_archive_linux}

Install OpenVINO™ Runtime on Linux from an Archive File
=========================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on the Linux operating
                 system, using an archive file.


.. note::

   Note that the Archive distribution:

   * offers both C/C++ and Python APIs
   * additionally includes code samples
   * is dedicated to Linux users (archives for other systems are also available)
   * may offer different hardware support under different operating systems
     (see the drop-down below for more details).

   .. dropdown:: Inference Options

      ===================  =====  =====  =====  =====
       Operating System     CPU    GPU    GNA    NPU
      ===================  =====  =====  =====  =====
       Debian9 armhf         V     n/a    n/a    n/a
       Ubuntu18 arm64        V     n/a    n/a    n/a
       CentOS7 x86_64        V      V     n/a    n/a
       Ubuntu18 x86_64       V      V      V     n/a
       Ubuntu20 x86_64       V      V      V      V
       Ubuntu22 x86_64       V      V      V      V
       RHEL8 x86_64          V      V      V     n/a
      ===================  =====  =====  =====  =====


.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <system_requirements>`

   .. tab-item:: Processor Notes
      :sync: processor-notes

      | To see if your processor includes the integrated graphics technology and supports iGPU inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__

   .. tab-item:: Software
      :sync: software

      * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
      * `Python 3.8 - 3.11, 64-bit <https://www.python.org/downloads/>`__
      * GCC:

      .. tab-set::

         .. tab-item:: Ubuntu 20.04
            :sync: ubuntu-20

            * GCC 9.3.0

         .. tab-item:: Ubuntu 18.04
            :sync: ubuntu-18

            * GCC 7.5.0

         .. tab-item:: RHEL 8
            :sync: rhel-8

            * GCC 8.4.1

         .. tab-item:: CentOS 7
            :sync: centos-7

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

4. Download the `OpenVINO Runtime archive file for your system <https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/>`_, extract the files, rename the extracted folder and move it to the desired path:

   .. tab-set::

      .. tab-item:: x86_64
         :sync: x86-64

         .. tab-set::

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64.tgz --output openvino_2023.3.0.tgz
                  tar -xf openvino_2023.3.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0

            .. tab-item:: Ubuntu 20.04
               :sync: ubuntu-20

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz --output openvino_2023.3.0.tgz
                  tar -xf openvino_2023.3.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0


            .. tab-item:: Ubuntu 18.04
               :sync: ubuntu-18

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu18_2023.3.0.13775.ceeafaf64f3_x86_64.tgz --output openvino_2023.3.0.tgz
                  tar -xf openvino_2023.3.0.tgz
                  sudo mv l_openvino_toolkit_ubuntu18_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0

            .. tab-item:: RHEL 8
               :sync: rhel-8

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_rhel8_2023.3.0.13775.ceeafaf64f3_x86_64.tgz --output openvino_2023.3.0.tgz
                  tar -xf openvino_2023.3.0.tgz
                  sudo mv l_openvino_toolkit_rhel8_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0

            .. tab-item:: CentOS 7
               :sync: centos-7

               .. code-block:: sh

                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_centos7_2023.3.0.13775.ceeafaf64f3_x86_64.tgz --output openvino_2023.3.0.tgz
                  tar -xf openvino_2023.3.0.tgz
                  sudo mv l_openvino_toolkit_centos7_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3.0


      .. tab-item:: ARM 64-bit
         :sync: arm-64

         .. code-block:: sh


            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu18_2023.3.0.13775.ceeafaf64f3_arm64.tgz -O openvino_2023.3.0.tgz
            tar -xf openvino_2023.3.0.tgz
            sudo mv l_openvino_toolkit_ubuntu18_2023.3.0.13775.ceeafaf64f3_arm64 /opt/intel/openvino_2023.3.0

      .. tab-item:: ARM 32-bit
         :sync: arm-32

         .. code-block:: sh

            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_debian9_2023.3.0.13775.ceeafaf64f3_armhf.tgz -O openvino_2023.3.0.tgz
            tar -xf openvino_2023.3.0.tgz
            sudo mv l_openvino_toolkit_debian9_2023.3.0.13775.ceeafaf64f3_armhf /opt/intel/openvino_2023.3.0


5. Install required system dependencies on Linux. To do this, OpenVINO provides a script in the extracted installation directory. Run the following command:

   .. code-block:: sh

      cd /opt/intel/openvino_2023.3.0
      sudo -E ./install_dependencies/install_openvino_dependencies.sh

6. (Optional) Install *numpy* Python Library:

   .. note::

      This step is required only when you decide to use Python API.

   You can use the ``requirements.txt`` file from the ``/opt/intel/openvino_2023.3.0/python`` folder:

   .. code-block:: sh

      cd /opt/intel/openvino_2023.3.0
      python3 -m pip install -r ./python/requirements.txt

7. For simplicity, it is useful to create a symbolic link as below:

   .. code-block:: sh

      cd /opt/intel

      sudo ln -s openvino_2023.3.0 openvino_2023

   .. note::
      If you have already installed a previous release of OpenVINO 2023, a symbolic link to the ``openvino_2023`` folder may already exist.
      Unlink the previous link with ``sudo unlink openvino_2023``, and then re-run the command above.


Congratulations, you have finished the installation! For some use cases you may still
need to install additional components. Check the description below, as well as the
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.

The ``/opt/intel/openvino_2023`` folder now contains the core components for OpenVINO.
If you used a different path in Step 2, for example, ``/home/<USER>/intel/``,
OpenVINO is now in ``/home/<USER>/intel/openvino_2023``. The path to the ``openvino_2023``
directory is also referred as ``<INSTALL_DIR>`` throughout the OpenVINO documentation.


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

The environment variables are set.




What's Next?
############################################################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

.. tab-set::

   .. tab-item:: Get started with Python
      :sync: get-started-py

      Try the `Python Quick Start Example <notebooks/201-vision-monodepth-with-output.html>`_
      to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

      .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
         :width: 400

      Visit the :doc:`Tutorials <tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

      * `OpenVINO Python API Tutorial <notebooks/002-openvino-api-with-output.html>`__
      * `Basic image classification program with Hello Image Classification <notebooks/001-hello-world-with-output.html>`__
      * `Convert a PyTorch model and use it for image background removal <notebooks/205-vision-background-removal-with-output.html>`__


   .. tab-item:: Get started with C++
      :sync: get-started-cpp

      Try the :doc:`C++ Quick Start Example <openvino_docs_get_started_get_started_demos>` for step-by-step instructions
      on building and running a basic image classification C++ application.

      .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
         :width: 400

      Visit the :doc:`Samples <openvino_docs_OV_UG_Samples_Overview>` page for other C++ example applications to get you started with OpenVINO, such as:

      * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_sample_hello_reshape_ssd.html>`__
      * `Object classification sample <openvino_sample_hello_classification.html>`__



Uninstalling the Intel® Distribution of OpenVINO™ Toolkit
###########################################################

If you have installed OpenVINO Runtime from archive files, you can uninstall it by deleting the archive files and the extracted folders.
Uninstallation removes all Intel® Distribution of OpenVINO™ Toolkit component files but does not affect user files in the installation directory.

If you have created the symbolic link, remove the link first:

.. code-block:: sh

   sudo rm /opt/intel/openvino_2023

To delete the files:

.. code-block:: sh

   rm -r <extracted_folder> && rm <path_to_archive>






Additional Resources
###########################################################

* :doc:`Troubleshooting Guide for OpenVINO Installation & Configuration <openvino_docs_get_started_guide_troubleshooting>`
* Converting models for use with OpenVINO™: :doc:`Convert a Model <openvino_docs_model_processing_introduction>`
* Writing your own OpenVINO™ applications: :doc:`OpenVINO™ Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`
* Sample applications: :doc:`OpenVINO™ Toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`
* Pre-trained deep learning models: :doc:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model_zoo>`
* IoT libraries and code samples in the GitHub repository: `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__




