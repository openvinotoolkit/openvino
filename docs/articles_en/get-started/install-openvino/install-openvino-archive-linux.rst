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

      ===================  =====  =====  =====
       Operating System     CPU    GPU    NPU
      ===================  =====  =====  =====
       Debian9 armhf         V     n/a    n/a
       Ubuntu20 arm64        V     n/a    n/a
       CentOS7 x86_64        V      V     n/a
       Ubuntu20 x86_64       V      V      V
       Ubuntu22 x86_64       V      V      V
       RHEL8 x86_64          V      V     n/a
      ===================  =====  =====  =====

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.


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

4. Download the `OpenVINO Runtime archive file for your system <https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/>`_, extract the files, rename the extracted folder and move it to the desired path:

   .. tab-set::

      .. tab-item:: x86_64
         :sync: x86-64

         .. tab-set::

            .. tab-item:: Ubuntu 24.04
               :sync: ubuntu-24

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_ubuntu24_2025.0.0.17942.1f68be9f594_x86_64.tgz --output openvino_2025.0.0.tgz
                  tar -xf openvino_2025.0.0.tgz
                  sudo mv openvino_toolkit_ubuntu24_2025.0.0.17942.1f68be9f594_x86_64 /opt/intel/openvino_2025.0.0

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_ubuntu22_2025.0.0.17942.1f68be9f594_x86_64.tgz --output openvino_2025.0.0.tgz
                  tar -xf openvino_2025.0.0.tgz
                  sudo mv openvino_toolkit_ubuntu22_2025.0.0.17942.1f68be9f594_x86_64 /opt/intel/openvino_2025.0.0

            .. tab-item:: Ubuntu 20.04
               :sync: ubuntu-20

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_ubuntu20_2025.0.0.17942.1f68be9f594_x86_64.tgz --output openvino_2025.0.0.tgz
                  tar -xf openvino_2025.0.0.tgz
                  sudo mv openvino_toolkit_ubuntu20_2025.0.0.17942.1f68be9f594_x86_64 /opt/intel/openvino_2025.0.0

            .. tab-item:: RHEL 8
               :sync: rhel-8

               .. code-block:: sh


                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_rhel8_2025.0.0.17942.1f68be9f594_x86_64.tgz --output openvino_2025.0.0.tgz
                  tar -xf openvino_2025.0.0.tgz
                  sudo mv openvino_toolkit_rhel8_2025.0.0.17942.1f68be9f594_x86_64 /opt/intel/openvino_2025.0.0

            .. tab-item:: CentOS 7
               :sync: centos-7

               .. code-block:: sh

                  curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_centos7_2025.0.0.17942.1f68be9f594_x86_64.tgz --output openvino_2025.0.0.tgz
                  tar -xf openvino_2025.0.0.tgz
                  sudo mv openvino_toolkit_centos7_2025.0.0.17942.1f68be9f594_x86_64 /opt/intel/openvino_2025.0.0


      .. tab-item:: ARM 64-bit
         :sync: arm-64

         .. code-block:: sh

            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_ubuntu20_2025.0.0.17942.1f68be9f594_arm64.tgz -O openvino_2025.0.0.tgz
            tar -xf openvino_2025.0.0.tgz
            sudo mv openvino_toolkit_ubuntu20_2025.0.0.17942.1f68be9f594_arm64 /opt/intel/openvino_2025.0.0

      .. tab-item:: ARM 32-bit
         :sync: arm-32

         .. code-block:: sh

            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_toolkit_debian10_2025.0.0.17942.1f68be9f594_armhf.tgz -O openvino_2025.0.0.tgz
            tar -xf openvino_2025.0.0.tgz
            sudo mv openvino_toolkit_debian10_2025.0.0.17942.1f68be9f594_armhf /opt/intel/openvino_2025.0.0


5. Install required system dependencies on Linux. To do this, OpenVINO provides a script in the extracted installation directory. Run the following command:

   .. code-block:: sh

      cd /opt/intel/openvino_2025.0.0
      sudo -E ./install_dependencies/install_openvino_dependencies.sh

6. (Optional) Install *numpy* Python Library:

   .. note::

      This step is required only when you decide to use Python API.

   You can use the ``requirements.txt`` file from the ``/opt/intel/openvino_2025.0.0/python`` folder:

   .. code-block:: sh

      cd /opt/intel/openvino_2025.0.0
      python3 -m pip install -r ./python/requirements.txt

7. For simplicity, it is useful to create a symbolic link as below:

   .. code-block:: sh

      cd /opt/intel

      sudo ln -s openvino_2025.0.0 openvino_2025

   .. note::
      If you have already installed a previous release of OpenVINO 2025, a symbolic link to the ``openvino_2025`` folder may already exist.
      Unlink the previous link with ``sudo unlink openvino_2025``, and then re-run the command above.


Congratulations, you have finished the installation! For some use cases you may still
need to install additional components. Check the description below, as well as the
:doc:`list of additional configurations <./configurations>`
to see if your case needs any of them.

The ``/opt/intel/openvino_2025`` folder now contains the core components for OpenVINO.
If you used a different path in Step 2, for example, ``/home/<USER>/intel/``,
OpenVINO is now in ``/home/<USER>/intel/openvino_2025``. The path to the ``openvino_2025``
directory is also referred as ``<INSTALL_DIR>`` throughout the OpenVINO documentation.


Step 2: Configure the Environment
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You must update several environment variables before you can compile and run OpenVINO applications.
Open a terminal window and run the ``setupvars.sh`` script as shown below to temporarily set your environment variables.
If your <INSTALL_DIR> is not ``/opt/intel/openvino_2025``, use the correct one instead.

.. code-block:: sh

   source /opt/intel/openvino_2025/setupvars.sh


If you have more than one OpenVINO version installed on your system, you can easily switch versions by sourcing the `setupvars.sh` of your choice.

.. note::

   The above command must be re-run every time you start a new terminal session.
   To set up Linux to automatically run the command every time a new terminal is opened,
   open ``~/.bashrc`` in your favorite editor and add ``source /opt/intel/openvino_2025/setupvars.sh`` after the last line.
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

      Try the `Python Quick Start Example <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth>`__
      to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

      .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
         :width: 400

      Visit the :doc:`Tutorials <../../../get-started/learn-openvino/interactive-tutorials-python>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

      * `OpenVINO Python API Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api>`__
      * `Basic image classification program with Hello Image Classification <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world>`__
      * `Convert a PyTorch model and use it for image background removal <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal>`__


   .. tab-item:: Get started with C++
      :sync: get-started-cpp

      Try the :doc:`C++ Quick Start Example <../../../get-started/learn-openvino/openvino-samples/get-started-demos>` for step-by-step instructions
      on building and running a basic image classification C++ application.

      .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
         :width: 400

      Visit the :doc:`Samples <../../../get-started/learn-openvino/openvino-samples>` page for other C++ example applications to get you started with OpenVINO, such as:

      * :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd>`
      * :doc:`Object classification sample <../../../get-started/learn-openvino/openvino-samples/hello-classification>`



Uninstalling the Intel® Distribution of OpenVINO™ Toolkit
###########################################################

If you have installed OpenVINO Runtime from archive files, you can uninstall it by deleting the archive files and the extracted folders.
Uninstallation removes all Intel® Distribution of OpenVINO™ Toolkit component files but does not affect user files in the installation directory.

If you have created the symbolic link, remove the link first:

.. code-block:: sh

   sudo rm /opt/intel/openvino_2025

To delete the files:

.. code-block:: sh

   rm -r <extracted_folder> && rm <path_to_archive>






Additional Resources
###########################################################

* :doc:`Troubleshooting Guide for OpenVINO Installation & Configuration <./configurations/troubleshooting-install-config>`
* Converting models for use with OpenVINO™: :doc:`Convert a Model <../../../openvino-workflow/model-preparation>`
* Writing your own OpenVINO™ applications: :doc:`OpenVINO™ Runtime User Guide <../../../openvino-workflow/running-inference>`
* Sample applications: :doc:`OpenVINO™ Toolkit Samples Overview <../../../get-started/learn-openvino/openvino-samples>`
* Pre-trained deep learning models on `Hugging Face <https://huggingface.co/OpenVINO>`__.
