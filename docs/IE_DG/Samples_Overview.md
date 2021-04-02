# Inference Engine Samples {#openvino_docs_IE_DG_Samples_Overview}

The Inference Engine sample applications are simple console applications that show how to utilize specific Inference Engine capabilities within an application, assist developers in executing specific tasks such as loading a model, running inference, querying specific device capabilities and etc. 

After installation of Intel® Distribution of OpenVINO™ toolkit, С, C++ and Python* sample applications are available in the following directories, respectively:
* `<INSTALL_DIR>/inference_engine/samples/c`
* `<INSTALL_DIR>/inference_engine/samples/cpp`
* `<INSTALL_DIR>/inference_engine/samples/python` 

Inference Engine sample applications include the following:
- **[Automatic Speech Recognition C++ Sample](../../inference-engine/samples/speech_sample/README.md)** – Acoustic model inference based on Kaldi neural networks and speech feature vectors.
- **Benchmark Application** – Estimates deep learning inference performance on supported devices for synchronous and asynchronous modes.
   - [Benchmark C++ Application](../../inference-engine/samples/benchmark_app/README.md) 
   - [Benchmark Python Application](../../inference-engine/tools/benchmark_tool/README.md)
- **Hello Classification Sample** – Inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs and supports Unicode paths).
   - [Hello Classification C++ Sample](../../inference-engine/samples/hello_classification/README.md)
   - [Hello Classification C Sample](../../inference-engine/ie_bridges/c/samples/hello_classification/README.md)
   - [Hello Classification Python Sample](../../inference-engine/ie_bridges/python/sample/hello_classification/README.md)
- **Hello NV12 Input Classification Sample** – Input of any size and layout can be provided to an infer request. The sample transforms the input to the NV12 color format and pre-process it automatically during inference. The sample supports only images as inputs. 
   - [Hello NV12 Input Classification C++ Sample](../../inference-engine/samples/hello_nv12_input_classification/README.md)
   - [Hello NV12 Input Classification C Sample](../../inference-engine/ie_bridges/c/samples/hello_nv12_input_classification/README.md)
- **Hello Query Device Sample** – Query of available Inference Engine devices and their metrics, configuration values.
   - [Hello Query Device C++ Sample](../../inference-engine/samples/hello_query_device/README.md)
   - [Hello Query Device Python* Sample](../../inference-engine/ie_bridges/python/sample/hello_query_device/README.md)
- **Hello Reshape SSD Sample** – Inference of SSD networks resized by ShapeInfer API according to an input size.
   - [Hello Reshape SSD C++ Sample**](../../inference-engine/samples/hello_reshape_ssd/README.md)
   - [Hello Reshape SSD Python Sample**](../../inference-engine/ie_bridges/python/sample/hello_reshape_ssd/README.md)
- **Image Classification Sample Async** – Inference of image classification networks like AlexNet and GoogLeNet using Asynchronous Inference Request API (the sample supports only images as inputs). 
   - [Image Classification C++ Sample Async](../../inference-engine/samples/classification_sample_async/README.md)
   - [Image Classification Python* Sample Async](../../inference-engine/ie_bridges/python/sample/classification_sample_async/README.md)
- **Neural Style Transfer Sample** – Style Transfer sample (the sample supports only images as inputs).
   - [Neural Style Transfer C++ Sample](../../inference-engine/samples/style_transfer_sample/README.md)
   - [Neural Style Transfer Python* Sample](../../inference-engine/ie_bridges/python/sample/style_transfer_sample/README.md)
- **nGraph Function Creation Sample** – Construction of the LeNet network using the nGraph function creation sample.
   - [nGraph Function Creation C++ Sample](../../inference-engine/samples/ngraph_function_creation_sample/README.md)
   - [nGraph Function Creation Python Sample](../../inference-engine/ie_bridges/python/sample/ngraph_function_creation_sample/README.md)
- **Object Detection for SSD Sample** – Inference of object detection networks based on the SSD, this sample is simplified version that supports only images as inputs. 
   - [Object Detection for SSD C++ Sample](../../inference-engine/samples/object_detection_sample_ssd/README.md)
   - [Object Detection for SSD C Sample](../../inference-engine/ie_bridges/c/samples/object_detection_sample_ssd/README.md)
   - [Object Detection for SSD Python* Sample](../../inference-engine/ie_bridges/python/sample/object_detection_sample_ssd/README.md)
 
> **NOTE**: All samples support input paths containing only ASCII characters, except the Hello Classification Sample, that supports Unicode.

## Media Files Available for Samples

To run the sample applications, you can use images and videos from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

## Samples that Support Pre-Trained Models

To run the sample, you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).

## Build the Sample Applications

### <a name="build_samples_linux"></a>Build the Sample Applications on Linux*

The officially supported Linux* build environment is the following:

* Ubuntu* 18.04 LTS 64-bit or CentOS* 7.6 64-bit
* GCC* 7.5.0 (for Ubuntu* 18.04) or GCC* 4.8.5 (for CentOS* 7.6)
* CMake* version 3.10 or higher

> **NOTE**: For building samples from the open-source version of OpenVINO™ toolkit, see the [build instructions on GitHub](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

To build the C or C++ sample applications for Linux, go to the `<INSTALL_DIR>/inference_engine/samples/c` or `<INSTALL_DIR>/inference_engine/samples/cpp` directory, respectively, and run the `build_samples.sh` script:
```sh
build_samples.sh
```

Once the build is completed, you can find sample binaries in the following folders:
* C samples: `~/inference_engine_c_samples_build/intel64/Release`
* C++ samples: `~/inference_engine_cpp_samples_build/intel64/Release`

You can also build the sample applications manually:

> **NOTE**: If you have installed the product as a root user, switch to root mode before you continue: `sudo -i`

1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named `build`:
```sh
mkdir build
```
> **NOTE**: If you ran the Image Classification verification script during the installation, the C++ samples build directory was already created in your home directory: `~/inference_engine_samples_build/`

2. Go to the created directory:
```sh
cd build
```

3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:
  - For release configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/inference_engine/samples/cpp
  ```
  - For debug configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/inference_engine/samples/cpp
  ```
4. Run `make` to build the samples:
```sh
make
```

For the release configuration, the sample application binaries are in `<path_to_build_directory>/intel64/Release/`;
for the debug configuration — in `<path_to_build_directory>/intel64/Debug/`.

### <a name="build_samples_windows"></a>Build the Sample Applications on Microsoft Windows* OS

The recommended Windows* build environment is the following:
* Microsoft Windows* 10
* Microsoft Visual Studio* 2017, or 2019
* CMake* version 3.10 or higher

> **NOTE**: If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.

To build the C or C++ sample applications on Windows, go to the `<INSTALL_DIR>\inference_engine\samples\c` or `<INSTALL_DIR>\inference_engine\samples\cpp` directory, respectively, and run the `build_samples_msvc.bat` batch file:
```sh
build_samples_msvc.bat
```

By default, the script automatically detects the highest Microsoft Visual Studio version installed on the machine and uses it to create and build
a solution for a sample code. Optionally, you can also specify the preferred Microsoft Visual Studio version to be used by the script. Supported
versions are `VS2017` and `VS2019`. For example, to build the C++ samples using the Microsoft Visual Studio 2017, use the following command:
```sh
<INSTALL_DIR>\inference_engine\samples\cpp\build_samples_msvc.bat VS2017
```

Once the build is completed, you can find sample binaries in the following folders:
* C samples: `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_c_samples_build\intel64\Release`
* C++ samples: `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_cpp_samples_build\intel64\Release`

You can also build a generated solution manually. For example, if you want to build C++ sample binaries in Debug configuration, run the appropriate version of the
Microsoft Visual Studio and open the generated solution file from the `C:\Users\<user>\Documents\Intel\OpenVINO\inference_engine_cpp_samples_build\Samples.sln`
directory.

### <a name="build_samples_macos"></a>Build the Sample Applications on macOS*

The officially supported macOS* build environment is the following:

* macOS* 10.15 64-bit
* Clang* compiler from Xcode* 10.1 or higher
* CMake* version 3.13 or higher

> **NOTE**: For building samples from the open-source version of OpenVINO™ toolkit, see the [build instructions on GitHub](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

To build the C or C++ sample applications for macOS, go to the `<INSTALL_DIR>/inference_engine/samples/c` or `<INSTALL_DIR>/inference_engine/samples/cpp` directory, respectively, and run the `build_samples.sh` script:
```sh
build_samples.sh
```

Once the build is completed, you can find sample binaries in the following folders:
* C samples: `~/inference_engine_c_samples_build/intel64/Release`
* C++ samples: `~/inference_engine_cpp_samples_build/intel64/Release`

You can also build the sample applications manually:

> **NOTE**: If you have installed the product as a root user, switch to root mode before you continue: `sudo -i`

> **NOTE**: Before proceeding, make sure you have OpenVINO™ environment set correctly. This can be done manually by
```sh
cd <INSTALL_DIR>/bin
source setupvars.sh
```

1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named `build`:
```sh
mkdir build
```
> **NOTE**: If you ran the Image Classification verification script during the installation, the C++ samples build directory was already created in your home directory: `~/inference_engine_samples_build/`

2. Go to the created directory:
```sh
cd build
```

3. Run CMake to generate the Make files for release or debug configuration. For example, for C++ samples:
  - For release configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Release <INSTALL_DIR>/inference_engine/samples/cpp
  ```
  - For debug configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Debug <INSTALL_DIR>/inference_engine/samples/cpp
  ```
4. Run `make` to build the samples:
```sh
make
```

For the release configuration, the sample application binaries are in `<path_to_build_directory>/intel64/Release/`;
for the debug configuration — in `<path_to_build_directory>/intel64/Debug/`.

## Get Ready for Running the Sample Applications

### Get Ready for Running the Sample Applications on Linux*

Before running compiled binary files, make sure your application can find the
Inference Engine and OpenCV libraries.
Run the `setupvars` script to set all necessary environment variables:
```sh
source <INSTALL_DIR>/bin/setupvars.sh
```

**(Optional)**: The OpenVINO environment variables are removed when you close the
shell. As an option, you can permanently set the environment variables as follows:

1. Open the `.bashrc` file in `<user_home_directory>`:
```sh
vi <user_home_directory>/.bashrc
```

2. Add this line to the end of the file:
```sh
source /opt/intel/openvino/bin/setupvars.sh
```

3. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.
4. To test your change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

You are ready to run sample applications. To learn about how to run a particular
sample, read the sample documentation by clicking the sample name in the samples
list above.

### Get Ready for Running the Sample Applications on Windows*

Before running compiled binary files, make sure your application can find the
Inference Engine and OpenCV libraries.
Use the `setupvars` script, which sets all necessary environment variables:
```sh
<INSTALL_DIR>\bin\setupvars.bat
```

To debug or run the samples on Windows in Microsoft Visual Studio, make sure you
have properly configured **Debugging** environment settings for the **Debug**
and **Release** configurations. Set correct paths to the OpenCV libraries, and
debug and release versions of the Inference Engine libraries.
For example, for the **Debug** configuration, go to the project's
**Configuration Properties** to the **Debugging** category and set the `PATH`
variable in the **Environment** field to the following:

```sh
PATH=<INSTALL_DIR>\deployment_tools\inference_engine\bin\intel64\Debug;<INSTALL_DIR>\opencv\bin;%PATH%
```
where `<INSTALL_DIR>` is the directory in which the OpenVINO toolkit is installed.

You are ready to run sample applications. To learn about how to run a particular
sample, read the sample documentation by clicking the sample name in the samples
list above.

## See Also
* [Introduction to Inference Engine](inference_engine_intro.md)
