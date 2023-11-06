# Install OpenVINO™ Runtime on Windows from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_windows}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows operating 
                 system, using an archive file.


.. note::
   
   Note that the Archive distribution:
   
   * offers both C/C++ and Python APIs
   * additionally includes code samples 
   * is dedicated to Windows users (archives for other systems are also available)


System Requirements
####################

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
   
      * `Microsoft Visual Studio 2019 with MSBuild <https://visualstudio.microsoft.com/vs/older-downloads/>`__ or `Microsoft Visual Studio 2022 <http://visualstudio.microsoft.com/  downloads/>`__
      * `CMake 3.14 or higher, 64-bit <https://cmake.org/download/>`__ (optional, only required for building sample applications)
      * `Python 3.8 - 3.11, 64-bit <https://www.python.org/downloads/windows/>`__
   
      .. note::
   
         To install Microsoft Visual Studio 2019, follow the `Microsoft Visual Studio installation guide <https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019>`__. You can choose to download the Community version. During installation in the **Workloads** tab, choose **Desktop development with C++**.
   
      .. note::
   
         You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.
   
      .. important::
   
          When installing Python, make sure you click the option **Add Python 3.x to PATH** to `add Python <https://docs.python.org/3/using/windows.html#installation-steps>`__ to your `PATH` environment variable.
   


Installing OpenVINO Runtime
###########################

.. _install-openvino-archive-windows:

Step 1: Download and Install OpenVINO Core Components
+++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Create an ``Intel`` folder in the ``C:\Program Files (x86)\`` directory. Skip this step if the folder already exists.

   You can also do this via command-lines. Open a new command prompt window as administrator by right-clicking **Command Prompt** from the Start menu and select **Run as administrator**, and then run the following command:

   .. code-block:: sh

      mkdir "C:\Program Files (x86)\Intel"


   .. note::

      ``C:\Program Files (x86)\Intel`` is the recommended folder. You may also use a different path if desired or if you don't have administrator privileges on your computer.


2. Download the `OpenVINO Runtime archive file for Windows <https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/>`__ to your local ``Downloads`` folder.

   If you prefer using command-lines, run the following commands in the command prompt window you opened:

   .. code-block:: sh

      cd <user_home>/Downloads
      curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/windows/w_openvino_toolkit_windows_2023.1.0.12185.47b736f63ed_x86_64.zip --output openvino_2023.1.0.zip


   .. note::

      A ``.sha256`` file is provided together with the archive file to validate your download process. To do that, download the ``.sha256`` file from the same repository and run ``CertUtil -hashfile openvino_2023.1.0.zip SHA256``. Compare the returned value in the output with what's in the ``.sha256`` file: if the values are the same, you have downloaded the correct file successfully; if not, create a Support ticket `here <https://www.intel.com/content/www/us/en/support/contact-intel.html>`__.


3. Use your favorite tool to extract the archive file, rename the extracted folder, and move it to the ``C:\Program Files (x86)\Intel`` directory.

   To do this step using command-line, run the following commands in the command prompt window you opened:

   .. code-block:: sh

      tar -xf openvino_2023.1.0.zip
      ren w_openvino_toolkit_windows_2023.1.0.12185.47b736f63ed_x86_64 openvino_2023.1.0
      move openvino_2023.1.0 "C:\Program Files (x86)\Intel"


4. (Optional) Install *numpy* Python Library:

   .. note::

      This step is required only when you decide to use Python API.

   You can use the ``requirements.txt`` file from the ``C:\Program Files (x86)\Intel\openvino_2023.1.0\python`` folder:

   .. code-block:: sh

      cd "C:\Program Files (x86)\Intel\openvino_2023.1.0"
      python -m pip install -r .\python\requirements.txt


5. For simplicity, it is useful to create a symbolic link. Open a command prompt window as administrator (see Step 1 for how to do this) and run the following commands:

   .. code-block:: sh

      cd C:\Program Files (x86)\Intel
      mklink /D openvino_2023 openvino_2023.1.0


   .. note::

      If you have already installed a previous release of OpenVINO 2022, a symbolic link to the ``openvino_2023`` folder may already exist. If you want to override it, navigate to the ``C:\Program Files (x86)\Intel`` folder and delete the existing linked folder before running the ``mklink`` command.


Congratulations, you have finished the installation! For some use cases you may still 
need to install additional components. Check the description below, as well as the 
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.

The ``C:\Program Files (x86)\Intel\openvino_2023`` folder now contains the core components for OpenVINO. 
If you used a different path in Step 1, you will find the ``openvino_2023`` folder there. 
The path to the ``openvino_2023`` directory is also referred as ``<INSTALL_DIR>`` 
throughout the OpenVINO documentation.



.. _set-the-environment-variables-windows:

Step 2: Configure the Environment
+++++++++++++++++++++++++++++++++

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the ``setupvars.bat`` batch file to temporarily set your environment variables. If your ``<INSTALL_DIR>`` is not ``C:\Program Files (x86)\Intel\openvino_2023``, use the correct directory instead.

.. code-block:: sh

   "C:\Program Files (x86)\Intel\openvino_2023\setupvars.bat"


.. important::

   The above command must be re-run every time a new Command Prompt window is opened.


.. note::

   If you see an error indicating Python is not installed, Python may not be added to the PATH environment variable 
   (as described `here <https://docs.python.org/3/using/windows.html#finding-the-python-executable>`__). 
   Check your system environment variables, and add Python if necessary.



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


.. _uninstall-from-windows:

Uninstalling OpenVINO Runtime
#############################

If you have installed OpenVINO Runtime from archive files, you can uninstall it by deleting the archive files and the extracted folders.
Uninstallation removes all Intel® Distribution of OpenVINO™ Toolkit component files but does not affect user files in the installation directory. 

If you have created the symbolic link, remove the link first.

Use either of the following methods to delete the files:

* Use Windows Explorer to remove the files.
* Open a Command Prompt and run:

.. code-block:: sh

   rmdir /s <extracted_folder>
   del <path_to_archive>






Additional Resources
####################

* :ref:`Troubleshooting Guide for OpenVINO Installation & Configuration <troubleshooting guide for install>`
* Converting models for use with OpenVINO™: :ref:`Model Optimizer Developer Guide <deep learning model optimizer>`
* Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning openvino runtime>`
* Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
* Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
* IoT libraries and code samples in the GitHUB repository: `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__

<!---
   To learn more about converting models from specific frameworks, go to: 
   * :ref:`Convert Your Caffe Model <convert model caffe>`
   * :ref:`Convert Your TensorFlow Model <convert model tf>`
   * :ref:`Convert Your TensorFlow Lite Model <convert model tfl>`
   * :ref:`Convert Your Apache MXNet Model <convert model mxnet>`
   * :ref:`Convert Your Kaldi Model <convert model kaldi>`
   * :ref:`Convert Your ONNX Model <convert model onnx>`
--->

@endsphinxdirective
