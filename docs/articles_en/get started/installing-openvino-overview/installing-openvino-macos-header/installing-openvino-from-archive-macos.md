# Install OpenVINO™ Runtime on macOS from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_macos}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on macOS operating 
                 system, using an archive file.


.. note::
   
   Note that the Archive distribution:
   
   * offers both C/C++ and Python APIs
   * additionally includes code samples 
   * is dedicated to macOS users (archives for other systems are also available)
   * is only supported for CPU Plugin


.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements
   
      | Full requirement listing is available in:
      | :doc:`System Requirements Page <system_requirements>`

   .. tab-item:: Software Requirements
      :sync: software-requirements
   
      * `CMake 3.13 or higher <https://cmake.org/download/>`__ (choose "macOS 10.13 or later"). Add ``/Applications/CMake.app/Contents/bin`` to path (for default install).
      * `Python 3.8 - 3.11 <https://www.python.org/downloads/mac-osx/>`__ (choose 3.8 - 3.11). Install and add to path.
      * Apple Xcode Command Line Tools. In the terminal, run ``xcode-select --install`` from any directory
      * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)


Installing OpenVINO Runtime
###########################

Step 1: Install OpenVINO Core Components
++++++++++++++++++++++++++++++++++++++++


1. Open a command prompt terminal window.
2. Create the ``/opt/intel`` folder for OpenVINO by using the following command. If the folder already exists, skip this command.

   .. code-block:: sh

      sudo mkdir /opt/intel


   .. note::

      The ``/opt/intel`` path is the recommended folder path for installing OpenVINO. You may use a different path if desired.


3. Browse to the current user's ``Downloads`` folder:

   .. code-block:: sh

      cd <user_home>/Downloads


4. Download the `OpenVINO Runtime archive file for macOS <https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/macos/>`__, extract the files, rename the extracted folder and move it to the desired path:

   .. tab-set::

      .. tab-item:: x86, 64-bit
         :sync: x86-64
   
         .. code-block:: sh
   
            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/macos/m_openvino_toolkit_macos_10_15_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz --output openvino_2023.2.0.tgz
            tar -xf openvino_2023.2.0.tgz
            sudo mv m_openvino_toolkit_macos_10_15_2023.2.0.13089.cfd42bd2cb0_x86_64 /opt/intel/openvino_2023.2.0
   
      .. tab-item:: ARM, 64-bit
         :sync: arm-64
   
         .. code-block:: sh
   
            curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/macos/m_openvino_toolkit_macos_11_0_2023.2.0.13089.cfd42bd2cb0_arm64.tgz --output openvino_2023.2.0.tgz
            tar -xf openvino_2023.2.0.tgz
            sudo mv m_openvino_toolkit_macos_11_0_2023.2.0.13089.cfd42bd2cb0_arm64 /opt/intel/openvino_2023.2.0

5. (Optional) Install *numpy* Python Library:

   .. note::

      This step is required only when you decide to use Python API.

   You can use the ``requirements.txt`` file from the ``/opt/intel/openvino_2023.2.0/python`` folder:

   .. code-block:: sh

      cd /opt/intel/openvino_2023.2.0
      python3 -m pip install -r ./python/requirements.txt

6. For simplicity, it is useful to create a symbolic link as below:

   .. code-block:: sh

      sudo ln -s /opt/intel/openvino_2023.2.0 /opt/intel/openvino_2023 


   .. note::

      If you have already installed a previous release of OpenVINO 2023, a symbolic link to the ``openvino_2023`` folder may already exist. Unlink the previous link with ``sudo unlink openvino_2023``, and then re-run the command above.


Congratulations, you have finished the installation! For some use cases you may still 
need to install additional components. Check the description below, as well as the 
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.

The ``/opt/intel/openvino_2023`` folder now contains the core components for OpenVINO. 
If you used a different path in Step 2, for example, ``/home/<USER>/intel/``, 
OpenVINO is now in ``/home/<USER>/intel/openvino_2023``. The path to the ``openvino_2023`` 
directory is also referred as ``<INSTALL_DIR>`` throughout the OpenVINO documentation.


Step 2: Configure the Environment
+++++++++++++++++++++++++++++++++

You must update several environment variables before you can compile and run OpenVINO applications. Open a terminal window and run the ``setupvars.sh`` 
script as shown below to temporarily set your environment variables. If your ``<INSTALL_DIR>`` (the folder you used to install OpenVINO) is not 
the default ``/opt/intel/openvino_2023``, use the correct one instead.

.. code-block:: sh

   cd /opt/intel/openvino_2023 
   source /opt/intel/openvino_2023/setupvars.sh


If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing the ``setupvars.sh`` of your choice.

.. note::

   The above command must be re-run every time you start a new terminal session. To set up macOS to automatically run the command every time a new terminal is opened, open ``~/.zshrc`` in your favorite editor and add ``source /opt/intel/openvino_2023/setupvars.sh`` after the last line. Next time when you open a terminal, you will see ``[setupvars.sh] OpenVINO™ environment initialized``. Changing ``~/.zshrc`` is not recommended when you have multiple OpenVINO versions on your machine and want to switch among them.



What's Next?
####################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

.. tab-set::

   .. tab-item:: Get started with Python
      :sync: get-started-py
   
      Try the `Python Quick Start Example <notebooks/201-vision-monodepth-with-output.html>`__ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.
   
      .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
         :width: 400
   
      Visit the :ref:`Tutorials <notebook tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:
   
      * `OpenVINO Python API Tutorial <notebooks/002-openvino-api-with-output.html>`__
      * `Basic image classification program with Hello Image Classification <notebooks/001-hello-world-with-output.html>`__
      * `Convert a PyTorch model and use it for image background removal <notebooks/205-vision-background-removal-with-output.html>`__
   
   .. tab-item:: Get started with C++
      :sync: get-started-cpp
   
      Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step instructions on building and running a basic image classification C++ application.
   
      .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
         :width: 400
   
      Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:
   
      * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`_
      * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`_

Uninstalling Intel® Distribution of OpenVINO™ Toolkit
#####################################################

If you have installed OpenVINO Runtime from archive files, you can uninstall it by deleting the archive files and the extracted folders.
Uninstallation removes all Intel® Distribution of OpenVINO™ Toolkit component files but does not affect user files in the installation directory. 

If you have created the symbolic link, remove the link first:
    
.. code-block:: sh

   sudo rm /opt/intel/openvino_2023
    
To delete the files:
    
.. code-block:: sh

   rm -r <extracted_folder> && rm <path_to_archive>


Additional Resources
####################

* :ref:`Troubleshooting Guide for OpenVINO Installation & Configuration <troubleshooting guide for install>`
* Converting models for use with OpenVINO™: :ref:`Model Optimizer User Guide <deep learning model optimizer>`
* Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning openvino runtime>`
* Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
* Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
* IoT libraries and code samples in the GitHUB repository: `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__



@endsphinxdirective
