# OpenVINO Samples {#openvino_docs_OV_UG_Samples_Overview}

@sphinxdirective

.. _code samples:

.. toctree::
   :maxdepth: 1
   :hidden:
   
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
   openvino_inference_engine_samples_benchmark_app_README
   openvino_inference_engine_tools_benchmark_tool_README

@endsphinxdirective

The OpenVINO samples are simple console applications that show how to utilize specific OpenVINO API capabilities within an application. They can assist in executing specific tasks, such as loading a model, running inference, querying specific device capabilities, etc.

During installation of OpenVINO™ Runtime, sample applications for С, C++ and Python are created in the following directories:
* `<INSTALL_DIR>/samples/c`
* `<INSTALL_DIR>/samples/cpp`
* `<INSTALL_DIR>/samples/python`

The applications include:

- **Speech Sample** - Acoustic model inference based on Kaldi neural networks and speech feature vectors:
   - [Automatic Speech Recognition C++ Sample](../../samples/cpp/speech_sample/README.md)
   - [Automatic Speech Recognition Python Sample](../../samples/python/speech_sample/README.md)
- **Hello Classification Sample** – Inference of image classification networks, such as AlexNet and GoogLeNet, using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs and supports Unicode paths):
   - [Hello Classification C++ Sample](../../samples/cpp/hello_classification/README.md)
   - [Hello Classification C Sample](../../samples/c/hello_classification/README.md)
   - [Hello Classification Python Sample](../../samples/python/hello_classification/README.md)
- **Hello NV12 Input Classification Sample** – Input of any size and layout can be provided to an infer request. The sample transforms the input to the `NV12` color format and pre-process it automatically during inference. The sample supports only images as inputs:
   - [Hello NV12 Input Classification C++ Sample](../../samples/cpp/hello_nv12_input_classification/README.md)
   - [Hello NV12 Input Classification C Sample](../../samples/c/hello_nv12_input_classification/README.md)
- **Hello Query Device Sample** – Query of available OpenVINO devices and their metrics, configuration values:
   - [Hello Query Device C++ Sample](../../samples/cpp/hello_query_device/README.md)
   - [Hello Query Device Python Sample](../../samples/python/hello_query_device/README.md)
- **Hello Reshape SSD Sample** – Inference of SSD networks resized by ShapeInfer API according to an input size:
   - [Hello Reshape SSD C++ Sample**](../../samples/cpp/hello_reshape_ssd/README.md)
   - [Hello Reshape SSD Python Sample**](../../samples/python/hello_reshape_ssd/README.md)
- **Image Classification Sample Async** – Inference of image classification networks, such as AlexNet and GoogLeNet, using Asynchronous Inference Request API (the sample supports only images as inputs):
   - [Image Classification Async C++ Sample](../../samples/cpp/classification_sample_async/README.md)
   - [Image Classification Async Python Sample](../../samples/python/classification_sample_async/README.md)
- **OpenVINO Model Creation Sample** – Construction of the LeNet model using the OpenVINO model creation sample:
   - [OpenVINO Model Creation C++ Sample](../../samples/cpp/model_creation_sample/README.md)
   - [OpenVINO Model Creation Python Sample](../../samples/python/model_creation_sample/README.md)


- **Benchmark Application** – Estimates deep learning inference performance on supported devices for synchronous and asynchronous modes:
   - [Benchmark C++ Tool](../../samples/cpp/benchmark_app/README.md)
   
   Keep in mind that the Python version of the benchmark tool is currently available only through the [OpenVINO Development Tools installation](../install_guides/installing-model-dev-tools.md). It is not created in the samples directory but can be launched with the following command:
   ```sh
   benchmark_app -m <model> -i <input> -d <device>
   ```

   For more information, see the [Benchmark Python Tool](../../tools/benchmark_tool/README.md) documentation.

> **NOTE**: All C++ samples support input paths that contain only ASCII characters, except for the Hello Classification Sample (it supports Unicode).

## Media Files Available for Samples

To run the sample applications, use images and videos available online in the [test data storage](https://storage.openvinotoolkit.org/data/test_data).

## Samples that Support Pre-Trained Models

To run the sample, use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from Open Model Zoo. The models can be downloaded using [Model Downloader](@ref omz_tools_downloader).

## Build the Sample Applications

### <a name="build_samples_linux"></a>Build the Sample Applications on Linux

The supported Linux build environment includes:

* Ubuntu 18.04 LTS 64-bit or Ubuntu 20.04 LTS 64-bit,
* GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04),
* CMake version 3.10 or higher.

> **NOTE**: For building samples from the open source version of OpenVINO™ toolkit, see the [build instructions on GitHub](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

To build the C or C++ sample applications for Linux, go to the `<INSTALL_DIR>/samples/c` or `<INSTALL_DIR>/samples/cpp` directory, respectively, and run the `build_samples.sh` script:
```sh
build_samples.sh
```

Once the build has been completed, sample binaries can be found in the following folders:
* C samples -- `~/inference_engine_c_samples_build/intel64/Release`,
* C++ samples -- `~/inference_engine_cpp_samples_build/intel64/Release`.

Sample applications can also be built manually:

> **NOTE**: If you have installed the product as a root user, switch to root mode (`sudo -i`) before you continue.

1. Navigate to a directory with write access and create a folder for samples (for example, `build`):
```sh
mkdir build
```
> **NOTE**: If you run the Image Classification verification script during the installation, the C++ samples build directory is created in your home directory -- `~/inference_engine_cpp_samples_build/`.

2. Go to the newly created directory:
```sh
cd build
```

3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:
  - For release configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp
  ```
  - For debug configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/samples/cpp
  ```
4. Run `make` to build the samples:
```sh
make
```

The sample application binaries can be found in:
* The release configuration: `<path_to_build_directory>/intel64/Release/`,
* The debug configuration - `<path_to_build_directory>/intel64/Debug/`.

### <a name="build_samples_windows"></a>Build the Sample Applications on Microsoft Windows

The recommended Windows build environment includes:
* Microsoft Windows 10,
* Microsoft Visual Studio 2019,
* CMake version 3.10 or higher.

> **NOTE**: Microsoft Visual Studio 2019 requires installation of CMake 3.14 or higher.

To build C or C++ sample applications on Windows, go to the `<INSTALL_DIR>\samples\c` or `<INSTALL_DIR>\samples\cpp` directory, respectively, and run the `build_samples_msvc.bat` batch file:
```sh
build_samples_msvc.bat
```

By default, the script automatically detects the highest Microsoft Visual Studio version installed on the machine and uses it to build a solution for a sample code.

Once the build has been completed, sample binaries can be found in the following folders:
* C samples -- `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_c_samples_build\intel64\Release`,
* C++ samples -- `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_cpp_samples_build\intel64\Release`.

Generated solution can also be built manually. For example, to build C++ sample binaries in Debug configuration, run the suitable version of the
Microsoft Visual Studio and open the generated solution file from the `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_cpp_samples_build\Samples.sln`
directory.

### <a name="build_samples_macos"></a>Build the Sample Applications on macOS*

The supported macOS build environment includes:

* macOS 10.15 64-bit or higher,
* Clang compiler from Xcode 10.1 or higher,
* CMake version 3.13 or higher.

> **NOTE**: To learn how to build samples from the open-source version of OpenVINO™ toolkit, see the [build instructions on GitHub](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

To build C or C++ sample applications for macOS, go to the `<INSTALL_DIR>/samples/c` or `<INSTALL_DIR>/samples/cpp` directory, respectively, and run the `build_samples.sh` script:
```sh
build_samples.sh
```

Once the build has been completed, sample binaries can be found in the following folders:
* C samples -- `~/inference_engine_c_samples_build/intel64/Release`,
* C++ samples -- `~/inference_engine_cpp_samples_build/intel64/Release`.

The sample applications can also be built manually:

> **NOTE**: If the product has been installed as a root user, switch to root mode (`sudo -i`) before you continue. 

> **NOTE**: Before proceeding, make sure you have OpenVINO™ environment set correctly. This can be done manually by
```sh
cd <INSTALL_DIR>/
source setupvars.sh
```

1. Navigate to a directory with write access and create a folder for samples (for example, `build`):
```sh
mkdir build
```
> **NOTE**: If you ran the Image Classification verification script during the installation, the C++ samples build directory was already created in your home directory -- `~/inference_engine_cpp_samples_build/`.

2. Go to the newly created directory:
```sh
cd build
```

3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:
  - For release configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/samples/cpp
  ```
  - For debug configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/samples/cpp
  ```
4. Run `make` to build the samples:
```sh
make
```

The sample application binaries can be found in:
* The release configuration -- `<path_to_build_directory>/intel64/Release/`,
* The debug configuration -- `<path_to_build_directory>/intel64/Debug/`.

## Get Ready for Running the Sample Applications

### Get Ready for Running the Sample Applications on Linux

Before running compiled binary files, check if your application can find the
OpenVINO Runtime libraries.
Run the `setupvars` script to set all necessary environment variables:
```sh
source <INSTALL_DIR>/setupvars.sh
```

**Optional**: The OpenVINO environment variables are removed when you close the
shell.  Optionally, you can permanently set the environment variables as follows:

1. Open the `.bashrc` file in `<user_home_directory>`:
```sh
vi <user_home_directory>/.bashrc
```

2. Append the file with the following command-line:
```sh
source /opt/intel/openvino_2022/setupvars.sh
```

3. Press the **Esc** key, type `:wq` and press the **Enter** key to save and close the file.
4. To test change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.
 
Sample applications are ready to start. To learn how to run a particular
sample, go to the documentation by clicking the name of a chosen sample from the list above.

### Preparing to Run the Sample Applications on Windows

Before running compiled binary files, check if your application can find the
OpenVINO Runtime libraries.
Use the `setupvars` script, which sets all necessary environment variables:
```sh
<INSTALL_DIR>\setupvars.bat
```

To debug or run the samples on Windows in Microsoft Visual Studio, make sure you
have properly configured **Debugging** environment settings for the **Debug**
and **Release** configurations. Set correct paths to the OpenCV libraries, and
debug and release versions of the OpenVINO Runtime libraries.

For example, for the **Debug** configuration, go to 
**Configuration Properties** of the project, then select the **Debugging** category and set the `PATH`
variable in the **Environment** field to the following:

```sh
PATH=<INSTALL_DIR>\runtime\bin;%PATH%
```
where `<INSTALL_DIR>` is the installation directory of OpenVINO toolkit.

Now, sample applications are ready to run. To learn how to run a particular
sample, read the sample documentation by clicking the sample name in the samples

## Additional Resources
* [OpenVINO™ Runtime User Guide](openvino_intro.md)
