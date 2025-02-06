Get Started with Samples
========================


.. meta::
   :description: Learn the details on the workflow of Intel® Distribution of OpenVINO™
                 toolkit, and how to run inference, using provided code samples.


To use OpenVINO samples, install OpenVINO using one of the following distributions:

* Archive files (recommended) - :doc:`Linux <../../../get-started/install-openvino/install-openvino-archive-linux>` | :doc:`Windows <../../../get-started/install-openvino/install-openvino-archive-windows>` | :doc:`macOS <../../../get-started/install-openvino/install-openvino-archive-macos>`
* :doc:`APT <../../../get-started/install-openvino/install-openvino-apt>` or :doc:`YUM <../../../get-started/install-openvino/install-openvino-yum>` for Linux
* :doc:`Docker image <../../../get-started/install-openvino/install-openvino-docker-linux>`
* `Build from source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__

If you install OpenVINO Runtime via archive files, sample applications are created in the following directories:

* ``<INSTALL_DIR>/samples/python``
* ``<INSTALL_DIR>/samples/cpp``
* ``<INSTALL_DIR>/samples/c``

.. note::
   If you install OpenVINO without samples, you can still get them directly from `the OpenVINO repository <https://github.com/openvinotoolkit/openvino/>`__.

Before you build samples, refer to the :doc:`system requirements <../../../about-openvino/release-notes-openvino/system-requirements>` page and make sure that all the prerequisites have been installed. Next, you can perform the following steps:

1. :ref:`Build Samples <build-samples>`.
2. :ref:`Select a Sample <select-sample>`.
3. :ref:`Download a suitable model <download-model>`.
4. :ref:`Download media files used as input, if necessary <download-media>`.

Once you perform all the steps, you can :ref:`run inference with the chosen sample application <run-inference>` to see the results.

.. _build-samples:

Build the Sample Applications
================================

Select a sample you want to use from the :doc:`OpenVINO Samples <../openvino-samples>` page, and follow the instructions below to build it in your operating system.

.. note::

   Some samples may also require `OpenCV <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__ to run properly. Make sure to install it for use with vision-oriented samples.

Instructions below show how to build sample applications with CMake. If you are interested in building them from source, check the `build instructions on GitHub <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .

.. tab-set::

   .. tab-item:: Linux
      :sync: linux


      .. tab-set::

         .. tab-item:: Python
            :sync: python

            Each Python sample directory contains the ``requirements.txt`` file, which you must install before running the sample:

            .. code-block:: sh

               cd <INSTALL_DIR>/samples/python/<SAMPLE_DIR>
               python3 -m pip install -r ./requirements.txt

         .. tab-item:: C and C++
            :sync: cpp

            To build the C or C++ sample applications for Linux, go to the ``<INSTALL_DIR>/samples/c`` or ``<INSTALL_DIR>/samples/cpp`` directory, respectively, and run the ``build_samples.sh`` script:

            .. code-block:: sh

               build_samples.sh

            Once the build is completed, you can find sample binaries in the following folders:

            * C samples: ``~/openvino_c_samples_build/<architecture>/Release``
            * C++ samples: ``~/openvino_cpp_samples_build/<architecture>/Release`` where the <architecture> is the output of ``uname -m``, for example, ``intel64``, ``armhf``, or ``aarch64``.

            You can also build the sample applications manually:

            .. note::

               If you have installed the product as a root user, switch to root mode before you continue: ``sudo -i`` .

            1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named ``build``:

               .. code-block:: sh

                  mkdir build

               .. note::

                  If you ran the Image Classification verification script during the installation, the C++ samples build directory is created in your home directory: ``~/openvino_cpp_samples_build/``

            2. Go to the created directory:

               .. code-block:: sh

                  cd build

            3. Run CMake to generate the Make files for release configuration. For example, for C++ samples:

               .. code-block:: sh

                  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp


            4. Run ``make`` to build the samples:

               .. code-block:: sh

                  cmake --build . --parallel

            For the release configuration, the sample application binaries are in ``<path_to_build_directory>/<architecture>/Release/``;
            for the debug configuration — in ``<path_to_build_directory>/<architecture>/Debug/``.

   .. tab-item:: Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: Python
            :sync: python

            Each Python sample directory contains the ``requirements.txt`` file, which you must install before running the sample:

            .. code-block:: sh

               cd <INSTALL_DIR>\samples\python\<SAMPLE_DIR>
               python -m pip install -r requirements.txt

         .. tab-item:: C and C++
            :sync: c-cpp

            .. note::

               If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14 or higher.

            You can build the C or C++ sample applications on Windows with either PowerShell or Command Prompt.

            .. tab-set::

               .. tab-item:: PowerShell
                  :sync: powershell

                  To build Samples with PowerShell, run the following command:

                  .. code-block:: sh

                     & <path-to-build-samples-folder>/build_samples.ps1

               .. tab-item:: Command Prompt
                  :sync: cmd

                  To build Samples with CMD, go to the ``<INSTALL_DIR>\samples\c`` or ``<INSTALL_DIR>\samples\cpp`` directory, respectively, and run the ``build_samples_msvc.bat`` batch file:

                  .. code-block:: sh

                     build_samples_msvc.bat

            By default, the script automatically detects the highest Microsoft Visual Studio version installed on the system and uses it to create and build a solution for a sample code

            Once the build is completed, you can find sample binaries in the following folders:

            * C samples: ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_c_samples_build\<architecture>\Release``
            * C++ samples: ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_cpp_samples_build\<architecture>\Release`` where the <architecture> is the output of ``echo PROCESSOR_ARCHITECTURE%``, for example, ``intel64`` (AMD64), or ``arm64``.

            You can also build a generated solution manually. For example, if you want to build C++ sample binaries in Debug configuration, run the appropriate version of the Microsoft Visual Studio and open the generated solution file from the ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_cpp_samples_build\Samples.sln`` directory.

   .. tab-item:: macOS
      :sync: macos

      .. tab-set::

         .. tab-item:: Python
            :sync: python

            Each Python sample directory contains the ``requirements.txt`` file, which you must install before running the sample:

            .. code-block:: sh

               cd <INSTALL_DIR>/samples/python/<SAMPLE_DIR>
               python3 -m pip install -r ./requirements.txt

         .. tab-item:: C and C++
            :sync: cpp

            .. note::

               For building samples from the open-source version of OpenVINO toolkit, see the `build instructions on GitHub <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .

            To build the C or C++ sample applications for macOS, go to the ``<INSTALL_DIR>/samples/c`` or ``<INSTALL_DIR>/samples/cpp`` directory, respectively, and run the ``build_samples.sh`` script:

            .. code-block:: sh

               build_samples.sh

            Once the build is completed, you can find sample binaries in the following folders:

            * C samples: ``~/openvino_c_samples_build/<architecture>/Release``
            * C++ samples: ``~/openvino_cpp_samples_build/<architecture>/Release``

            You can also build the sample applications manually. Before proceeding, make sure you have OpenVINO™ environment set correctly. This can be done manually by:

            .. code-block:: sh

               cd <INSTALL_DIR>/
               source setupvars.sh

            .. note::

               If you have installed the product as a root user, switch to root mode before you continue: ``sudo -i``

            1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named ``build``:

               .. code-block:: sh

                  mkdir build

               .. note::

                  If you ran the Image Classification verification script during the installation, the C++ samples build directory was already created in your home directory: ``~/openvino_cpp_samples_build/``

            2. Go to the created directory:

               .. code-block:: sh

                  cd build

            3. Run CMake to generate the Make files for release configuration. For example, for C++ samples:

               .. code-block:: sh

                  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp


            4. Run ``make`` to build the samples:

               .. code-block:: sh

                  make

            For the release configuration, the sample application binaries are in ``<path_to_build_directory>/<architecture>/Release/``; for the debug configuration — in ``<path_to_build_directory>/<architecture>/Debug/``.


.. _select-sample:

Sample Application Setup
================================

First, select a sample from the :doc:`Sample Overview <../openvino-samples>` and read the dedicated article to learn how to run it.

.. _download-model:

Download the Models
--------------------

You need a model that is specific for your inference task. You can get it from one of model repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.


Convert the Model
--------------------

If Your model requires conversion, check the :doc:`article <../../../openvino-workflow/model-preparation>` for information how to do it.

.. _download-media:

Download a Media to use
-----------------------

Most of the samples require you to provide an image or a video as input for the model. OpenVINO provides several sample images and videos for you to run code samples and demo applications:

- `Sample images and video <https://storage.openvinotoolkit.org/data/test_data/>`__

To run the sample applications, you can use images and videos from the media files collection available `here <https://storage.openvinotoolkit.org/data/test_data>`__ . As an alternative, you can get them from sites like `Pexels <https://pexels.com>`__ or `Google Images <https://images.google.com>`__ .

.. _run-inference:

Run Inference on a Sample
================================

To run the code sample with an input image using the IR model:


1. Set up the OpenVINO environment variables:

   .. tab-set::

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            source  <INSTALL_DIR>/setupvars.sh

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            <INSTALL_DIR>\setupvars.bat

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            source <INSTALL_DIR>/setupvars.sh

.. note::

   OpenVINO environment variables can be set up by running the following command in PowerShell:

   .. code-block:: sh

      . <path-to-setupvars-folder>/setupvars.ps1

2. Go to the code samples release directory created when you built the samples earlier:

   .. tab-set::

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            cd ~/openvino_cpp_samples_build/intel64/Release

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            cd  %USERPROFILE%\Documents\Intel\OpenVINO\openvino_samples_build\intel64\Release

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            cd ~/openvino_cpp_samples_build/intel64/Release

3. Run the code sample executable, specifying the input media file, the IR for your model, and a target device for performing inference:


   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. tab-set::

            .. tab-item:: Linux
               :sync: linux

               .. code-block:: sh

                  python <sample.py file> -m <path_to_model> -i <path_to_media> -d <target_device>

            .. tab-item:: Windows
               :sync: windows

               .. code-block:: bat

                  python <sample.py file> -m <path_to_model> -i <path_to_media> -d <target_device>

            .. tab-item:: macOS
               :sync: macos

               .. code-block:: sh

                  python <sample.py file> -m <path_to_model> -i <path_to_media> -d <target_device>

      .. tab-item:: C++
         :sync: cpp

         .. tab-set::

            .. tab-item:: Linux
               :sync: linux

               .. code-block:: sh

                  <sample.exe file> -i <path_to_media> -m <path_to_model> -d <target_device>

            .. tab-item:: Windows
               :sync: windows

               .. code-block:: bat

                  <sample.exe file> -i <path_to_media> -m <path_to_model> -d <target_device>

            .. tab-item:: macOS
               :sync: macos

               .. code-block:: sh

                  <sample.exe file> -i <path_to_media> -m <path_to_model> -d <target_device>


Examples
================================

Running Inference on CPU
------------------------

The following command shows how to run the Image Classification Code Sample using the `dog.bmp <https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp>`__ file as an input image, the model in IR format from the ``ir`` directory, and the CPU as the target hardware:

.. note::

   * Running inference on Intel® Processor Graphics (GPU) requires :doc:`additional hardware configuration steps <../../../get-started/install-openvino/configurations/configurations-intel-gpu>`, as described earlier on this page.
   * Running on GPU is not compatible with macOS.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            .. code-block:: sh

               python classification_sample_async.py -m ~/ir/googlenet-v1.xml -i ~/Downloads/dog.bmp -d CPU

         .. tab-item:: Windows
            :sync: windows

            .. code-block:: bat

               python classification_sample_async.py -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -i %USERPROFILE%\Downloads\dog.bmp -d CPU

         .. tab-item:: macOS
            :sync: macos

            .. code-block:: sh

               python classification_sample_async.py -m ~/ir/googlenet-v1.xml -i ~/Downloads/dog.bmp -d CPU

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            .. code-block:: sh

               ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU

         .. tab-item:: Windows
            :sync: windows

            .. code-block:: bat

               .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d CPU

         .. tab-item:: macOS
            :sync: macos

            .. code-block:: sh

               ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU


When the sample application is complete, you are given the label and confidence for the top 10 categories. The input image and sample output of the inference results is shown below:

.. image:: ../../../assets/images/dog.png

.. code-block:: sh

   Top 10 results:

   Image dog.bmp

      classid probability label
      ------- ----------- -----
      156     0.6875963   Blenheim spaniel
      215     0.0868125   Brittany spaniel
      218     0.0784114   Welsh springer spaniel
      212     0.0597296   English setter
      217     0.0212105   English springer, English springer spaniel
      219     0.0194193   cocker spaniel, English cocker spaniel, cocker
      247     0.0086272   Saint Bernard, St Bernard
      157     0.0058511   papillon
      216     0.0057589   clumber, clumber spaniel
      154     0.0052615   Pekinese, Pekingese, Peke


Other Samples
================================

Articles in this section describe all sample applications provided with OpenVINO. They will give you more information on how each of them works, giving you a convenient starting point for your own application.

