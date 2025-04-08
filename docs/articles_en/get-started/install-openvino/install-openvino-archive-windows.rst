Install OpenVINO™ Runtime on Windows from an Archive File
===========================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows operating
                 system, using an archive file.


.. note::

   Note that the Archive distribution:

   * offers both C/C++ and Python APIs
   * additionally includes code samples
   * is dedicated to Windows users (archives for other systems are also available)

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.


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


2. Download the `OpenVINO Runtime archive file for Windows <https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/windows/>`__ to your local ``Downloads`` folder.

   If you prefer using command-lines, run the following commands in the command prompt window you opened:

   .. code-block:: sh

      cd <user_home>/Downloads
      curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/windows/openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64.zip --output openvino_2025.0.0.zip

   .. note::

      A ``.sha256`` file is provided together with the archive file to validate your download process. To do that, download the ``.sha256`` file from the same repository and run ``CertUtil -hashfile openvino_2025.0.0.zip SHA256``. Compare the returned value in the output with what's in the ``.sha256`` file: if the values are the same, you have downloaded the correct file successfully; if not, create a Support ticket `here <https://www.intel.com/content/www/us/en/support/contact-intel.html>`__.


3. Use your favorite tool to extract the archive file, rename the extracted folder, and move it to the ``C:\Program Files (x86)\Intel`` directory.

   To do this step using command-line, run the following commands in the command prompt window you opened:

   .. code-block:: sh

      tar -xf openvino_2025.0.0.zip
      ren openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64 openvino_2025.0.0
      move openvino_2025.0.0 "C:\Program Files (x86)\Intel"


4. (Optional) Install *numpy* Python Library:

   .. note::

      This step is required only when you decide to use Python API.

   You can use the ``requirements.txt`` file from the ``C:\Program Files (x86)\Intel\openvino_2025.0.0\python`` folder:

   .. code-block:: sh

      cd "C:\Program Files (x86)\Intel\openvino_2025.0.0"
      python -m pip install -r .\python\requirements.txt


5. For simplicity, it is useful to create a symbolic link. Open a command prompt window as administrator (see Step 1 for how to do this) and run the following commands:

   .. code-block:: sh

      cd C:\Program Files (x86)\Intel
      mklink /D openvino_2025 openvino_2025.0.0


   .. note::

      If you have already installed a previous release of OpenVINO 2025, a symbolic link to the ``openvino_2025`` folder may already exist. If you want to override it, navigate to the ``C:\Program Files (x86)\Intel`` folder and delete the existing linked folder before running the ``mklink`` command.


Congratulations, you have finished the installation! For some use cases you may still
need to install additional components. Check the description below, as well as the
:doc:`list of additional configurations <./configurations>`
to see if your case needs any of them.

The ``C:\Program Files (x86)\Intel\openvino_2025`` folder now contains the core components for OpenVINO.
If you used a different path in Step 1, you will find the ``openvino_2025`` folder there.
The path to the ``openvino_2025`` directory is also referred as ``<INSTALL_DIR>``
throughout the OpenVINO documentation.



.. _set-the-environment-variables-windows:

Step 2: Configure the Environment
+++++++++++++++++++++++++++++++++

You must update several environment variables before you can compile and run OpenVINO™ applications.

.. tab-set::

   .. tab-item:: PowerShell
      :sync: powershell

      Open the PowerShell, and run the ``setupvars.ps1`` file to temporarily set your environment variables.

      .. code-block:: sh

         . <path-to-setupvars-folder>/setupvars.ps1

   .. tab-item:: Command Prompt
      :sync: cmd

      Open the Command Prompt, and run the ``setupvars.bat`` batch file to temporarily set your environment variables.
      If your ``<INSTALL_DIR>`` is not ``C:\Program Files (x86)\Intel\openvino_2025``, use the correct directory instead.

      .. code-block:: sh

         "C:\Program Files (x86)\Intel\openvino_2025\setupvars.bat"

      .. important::

         You need to run the command for each new Command Prompt window.


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

      Try the `Python Quick Start Example <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth>`__ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

      .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
         :width: 400

      Visit the :doc:`Tutorials <../../../get-started/learn-openvino/interactive-tutorials-python>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

      * `OpenVINO Python API Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api>`__
      * `Basic image classification program with Hello Image Classification <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world>`__
      * `Convert a PyTorch model and use it for image background removal <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal>`__

   .. tab-item:: Get started with C++
      :sync: get-started-cpp

      Try the :doc:`C++ Quick Start Example <../../../get-started/learn-openvino/openvino-samples/get-started-demos>` for step-by-step instructions on building and running a basic image classification C++ application.

      .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
         :width: 400

      Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:

      * :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd>`
      * :doc:`Object classification sample <../../../get-started/learn-openvino/openvino-samples/hello-classification>`


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

* :doc:`Troubleshooting Guide for OpenVINO Installation & Configuration <../install-openvino>`
* :doc:`Convert models for use with OpenVINO™ <../../../openvino-workflow/model-preparation/convert-model-to-ir>`
* :doc:`Write your own OpenVINO™ applications <../../../openvino-workflow/running-inference>`
* Sample applications: :doc:`OpenVINO™ Toolkit Samples Overview <../../../get-started/learn-openvino/openvino-samples>`
* Pre-trained deep learning models on `Hugging Face <https://huggingface.co/OpenVINO>`__.
