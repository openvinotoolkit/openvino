# OpenVINO™ Samples {#openvino_docs_OV_UG_Samples_Overview}

@sphinxdirective

.. _code samples:

.. meta::
   :description: OpenVINO™ samples include a collection of simple console applications 
                 that explain how to implement the capabilities and features of 
                 OpenVINO API into an application.


.. toctree::
   :maxdepth: 1
   :hidden:
   
   Get Started with C++ Samples <openvino_docs_get_started_get_started_demos>
   openvino_inference_engine_samples_classification_sample_async_README
   openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README
   openvino_inference_engine_samples_hello_classification_README
   openvino_inference_engine_ie_bridges_c_samples_hello_classification_README
   openvino_inference_engine_ie_bridges_python_sample_hello_classification_README
   openvino_inference_engine_samples_hello_reshape_ssd_README
   openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README
   openvino_inference_engine_samples_hello_nv12_input_classification_README
   openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README
   openvino_inference_engine_samples_hello_query_device_README
   openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README
   openvino_inference_engine_samples_model_creation_sample_README
   openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README
   openvino_inference_engine_samples_speech_sample_README
   openvino_inference_engine_ie_bridges_python_sample_speech_sample_README
   openvino_inference_engine_samples_sync_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_sync_benchmark_README
   openvino_inference_engine_samples_throughput_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README
   openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README
   openvino_inference_engine_samples_benchmark_app_README
   openvino_inference_engine_tools_benchmark_tool_README


The OpenVINO™ samples are simple console applications that show how to utilize specific OpenVINO API capabilities within an application. They can assist you in executing specific tasks such as loading a model, running inference, querying specific device capabilities, etc.

If you installed OpenVINO Runtime via archive files, sample applications for С, and C++, and Python are created in the following directories:

* ``<INSTALL_DIR>/samples/c``
* ``<INSTALL_DIR>/samples/cpp``
* ``<INSTALL_DIR>/samples/python``

If you installed OpenVINO via PyPI, download `the OpenVINO repository <https://github.com/openvinotoolkit/openvino/>`__ and use samples from ``samples/python``.

The applications include:

- **Speech Sample** - ``[DEPRECATED]`` Acoustic model inference based on Kaldi neural networks and speech feature vectors.

  - :doc:`Automatic Speech Recognition C++ Sample <openvino_inference_engine_samples_speech_sample_README>`
  - :doc:`Automatic Speech Recognition Python Sample <openvino_inference_engine_ie_bridges_python_sample_speech_sample_README>`

- **Hello Classification Sample** – Inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs and supports Unicode paths).

  - :doc:`Hello Classification C++ Sample <openvino_inference_engine_samples_hello_classification_README>`
  - :doc:`Hello Classification C Sample <openvino_inference_engine_ie_bridges_c_samples_hello_classification_README>`
  - :doc:`Hello Classification Python Sample <openvino_inference_engine_ie_bridges_python_sample_hello_classification_README>`

- **Hello NV12 Input Classification Sample** – Input of any size and layout can be provided to an infer request. The sample transforms the input to the NV12 color format and pre-process it automatically during inference. The sample supports only images as inputs.

  - :doc:`Hello NV12 Input Classification C++ Sample <openvino_inference_engine_samples_hello_nv12_input_classification_README>`
  - :doc:`Hello NV12 Input Classification C Sample <openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README>`

- **Hello Query Device Sample** – Query of available OpenVINO devices and their metrics, configuration values.

  - :doc:`Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>`
  - :doc:`Hello Query Device Python* Sample <openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README>`

- **Hello Reshape SSD Sample** – Inference of SSD networks resized by ShapeInfer API according to an input size.

  - :doc:`Hello Reshape SSD C++ Sample** <openvino_inference_engine_samples_hello_reshape_ssd_README>`
  - :doc:`Hello Reshape SSD Python Sample** <openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README>`

- **Image Classification Sample Async** – Inference of image classification networks like AlexNet and GoogLeNet using Asynchronous Inference Request API (the sample supports only images as inputs).

  - :doc:`Image Classification Async C++ Sample <openvino_inference_engine_samples_classification_sample_async_README>`
  - :doc:`Image Classification Async Python* Sample <openvino_inference_engine_ie_bridges_python_sample_classification_sample_async_README>`

- **OpenVINO Model Creation Sample** – Construction of the LeNet model using the OpenVINO model creation sample.

  - :doc:`OpenVINO Model Creation C++ Sample <openvino_inference_engine_samples_model_creation_sample_README>`
  - :doc:`OpenVINO Model Creation Python Sample <openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README>`

- **Benchmark Samples** - Simple estimation of a model inference performance

  - :doc:`Sync Benchmark C++ Sample <openvino_inference_engine_samples_sync_benchmark_README>`
  - :doc:`Sync Benchmark Python* Sample <openvino_inference_engine_ie_bridges_python_sample_sync_benchmark_README>`
  - :doc:`Throughput Benchmark C++ Sample <openvino_inference_engine_samples_throughput_benchmark_README>`
  - :doc:`Throughput Benchmark Python* Sample <openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README>`
  - :doc:`Bert Benchmark Python* Sample <openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README>`

- **Benchmark Application** – Estimates deep learning inference performance on supported devices for synchronous and asynchronous modes.

  - :doc:`Benchmark C++ Tool <openvino_inference_engine_samples_benchmark_app_README>`  

    Note that the Python version of the benchmark tool is a core component of the OpenVINO installation package and 
    may be executed with the following command: ``benchmark_app -m <model> -i <input> -d <device>``. 
    For more information, check the :doc:`Benchmark Python Tool <openvino_inference_engine_tools_benchmark_tool_README>`.

.. note:: 
   
   All C++ samples support input paths containing only ASCII characters, except for the Hello Classification Sample, which supports Unicode.


Media Files Available for Samples
#################################

To run the sample applications, you can use images and videos from the media files collection available `here <https://storage.openvinotoolkit.org/data/test_data>`__ .

Samples that Support Pre-Trained Models
#######################################

To run the sample, you can use :doc:`public <omz_models_group_public>` or :doc:`Intel's <omz_models_group_intel>` pre-trained models from the Open Model Zoo. The models can be downloaded using the :doc:`Model Downloader <omz_tools_downloader>`.

Build the Sample Applications
#############################



Build the Sample Applications on Linux
++++++++++++++++++++++++++++++++++++++

The officially supported Linux build environment is the following:

* Ubuntu 18.04 LTS 64-bit or Ubuntu 20.04 LTS 64-bit
* GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
* CMake version 3.10 or higher

.. note::
   
   For building samples from the open-source version of OpenVINO toolkit, see the `build instructions on GitHub <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .

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
   
      If you run the Image Classification verification script during the installation, the C++ samples build directory is created in your home directory: ``~/openvino_cpp_samples_build/``
   
2. Go to the created directory:
   
   .. code-block:: sh
      
      cd build
   
3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:
   
   - For release configuration:

     .. code-block:: sh
      
        cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp
   
   - For debug configuration:

     .. code-block:: sh
        
        cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/samples/cpp

4. Run ``make`` to build the samples:

   .. code-block:: sh
      
      make

For the release configuration, the sample application binaries are in ``<path_to_build_directory>/<architecture>/Release/``;
for the debug configuration — in ``<path_to_build_directory>/<architecture>/Debug/``.

.. _build-samples-windows:

Build the Sample Applications on Microsoft Windows
++++++++++++++++++++++++++++++++++++++++++++++++++

The recommended Windows build environment is the following:

* Microsoft Windows 10
* Microsoft Visual Studio 2019
* CMake version 3.10 or higher

.. note:: 

   If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14 or higher.

To build the C or C++ sample applications on Windows, go to the ``<INSTALL_DIR>\samples\c`` or ``<INSTALL_DIR>\samples\cpp`` directory, respectively, and run the ``build_samples_msvc.bat`` batch file:

.. code-block:: sh
   
   build_samples_msvc.bat

By default, the script automatically detects the highest Microsoft Visual Studio version installed on the machine and uses it to create and build a solution for a sample code

Once the build is completed, you can find sample binaries in the following folders:

* C samples: ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_c_samples_build\<architecture>\Release``
* C++ samples: ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_cpp_samples_build\<architecture>\Release`` where the <architecture> is the output of ``echo PROCESSOR_ARCHITECTURE%``, for example, ``intel64`` (AMD64), or ``arm64``.

You can also build a generated solution manually. For example, if you want to build C++ sample binaries in Debug configuration, run the appropriate version of the Microsoft Visual Studio and open the generated solution file from the ``C:\Users\<user>\Documents\Intel\OpenVINO\openvino_cpp_samples_build\Samples.sln`` directory.

.. _build-samples-macos:

Build the Sample Applications on macOS
++++++++++++++++++++++++++++++++++++++

The officially supported macOS build environment is the following:

* macOS 10.15 64-bit or higher
* Clang compiler from Xcode 10.1 or higher
* CMake version 3.13 or higher

.. note:: 

   For building samples from the open-source version of OpenVINO toolkit, see the `build instructions on GitHub <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .

To build the C or C++ sample applications for macOS, go to the ``<INSTALL_DIR>/samples/c`` or ``<INSTALL_DIR>/samples/cpp`` directory, respectively, and run the ``build_samples.sh`` script:

.. code-block:: sh
   
   build_samples.sh

Once the build is completed, you can find sample binaries in the following folders:

* C samples: ``~/openvino_c_samples_build/<architecture>/Release``
* C++ samples: ``~/openvino_cpp_samples_build/<architecture>/Release``

You can also build the sample applications manually:

.. note::

   If you have installed the product as a root user, switch to root mode before you continue: ``sudo -i``

.. note:: 

   Before proceeding, make sure you have OpenVINO™ environment set correctly. This can be done manually by

.. code-block:: sh

   cd <INSTALL_DIR>/
   source setupvars.sh

1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named ``build``:

   .. code-block:: sh
   
      mkdir build
   
   .. note:: 
   
      If you ran the Image Classification verification script during the installation, the C++ samples build directory was already created in your home directory: ``~/openvino_cpp_samples_build/``
   
2. Go to the created directory:

   .. code-block:: sh
   
      cd build

3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:

   - For release configuration:

     .. code-block:: sh

        cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp
   
   - For debug configuration:

     .. code-block:: sh

        cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/samples/cpp
   
4. Run ``make`` to build the samples:

   .. code-block:: sh
   
      make

For the release configuration, the sample application binaries are in ``<path_to_build_directory>/<architecture>/Release/``; for the debug configuration — in ``<path_to_build_directory>/<architecture>/Debug/``.

Get Ready for Running the Sample Applications
#############################################

Get Ready for Running the Sample Applications on Linux
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Before running compiled binary files, make sure your application can find the OpenVINO Runtime libraries. Run the ``setupvars`` script to set all necessary environment variables:

.. code-block:: sh
   
   source <INSTALL_DIR>/setupvars.sh

(Optional) Set Environment Variables Permanently
------------------------------------------------

The OpenVINO environment variables are removed when you close the shell. As an option, you can permanently set the environment variables as follows:

1. Open the ``.bashrc`` file in ``<user_home_directory>``:

   .. code-block:: sh
      
      vi <user_home_directory>/.bashrc

2. Add this line to the end of the file:

   .. code-block:: sh
   
      source /opt/intel/openvino_2023/setupvars.sh

3. Save and close the file: press the **Esc** key, type ``:wq`` and press the **Enter** key.
4. To test your change, open a new terminal. You will see ``[setupvars.sh] OpenVINO environment initialized``.

You are ready to run sample applications. To learn about how to run a particular sample, read the sample documentation by clicking the sample name in the samples list above.

Get Ready for Running the Sample Applications on Windows
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Before running compiled binary files, make sure your application can find the OpenVINO Runtime libraries. Use the ``setupvars`` script, which sets all necessary environment variables:

.. code-block:: sh
   
   <INSTALL_DIR>\setupvars.bat

To debug or run the samples on Windows in Microsoft Visual Studio, make sure you have properly configured **Debugging** environment settings for the **Debug** and **Release** configurations. Set correct paths to the OpenCV libraries, and debug and release versions of the OpenVINO Runtime libraries. For example, for the **Debug** configuration, go to the project's **Configuration Properties** to the **Debugging** category and set the ``PATH`` variable in the **Environment** field to the following:

.. code-block:: sh

   PATH=<INSTALL_DIR>\runtime\bin;%PATH%

where ``<INSTALL_DIR>`` is the directory in which the OpenVINO toolkit is installed.

You are ready to run sample applications. To learn about how to run a particular sample, read the sample documentation by clicking the sample name in the samples list above.

See Also
########

* :doc:`OpenVINO Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

@endsphinxdirective

