# Build Inference Engine

## Contents

- [Introduction](#introduction)
- [Build on Linux Systems](#build-on-linux-systems)
  - [Software Requirements](#software-requirements)
  - [Build Steps](#build-steps)
  - [Additional Build Options](#additional-build-options)
- [Build for Raspbian Stretch OS](#build-for-raspbian-stretch-os)
  - [Hardware Requirements](#hardware-requirements)
  - [Native Compilation](#native-compilation)
  - [Cross Compilation Using Docker](#cross-compilation-using-docker)
  - [Additional Build Options](#additional-build-options-1)
- [Build on Windows Systems](#build-on-windows-systems)
  - [Software Requirements](#software-requirements-1)
  - [Build Steps](#build-steps-1)
  - [Additional Build Options](#additional-build-options-2)
  - [Building Inference Engine with Ninja Build System](#building-inference-engine-with-ninja-build-system)
- [Build on macOS Systems](#build-on-macos-systems)
  - [Software Requirements](#software-requirements-2)
  - [Build Steps](#build-steps-2)
  - [Additional Build Options](#additional-build-options-3)
- [Use Custom OpenCV Builds for Inference Engine](#use-custom-opencv-builds-for-inference-engine)
- <a href=#optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2>Additional Installation Steps for the Intel Movidius Neural Compute Stick and Neural Compute Stick 2</a>
  - [For Linux, Raspbian Stretch OS](#for-linux-raspbian-stretch-os)
  - [For Windows](#for-windows-1)
- [Next Steps](#next-steps)
- [Additional Resources](#additional-resources)

## Introduction
The Inference Engine can infer models in different formats with various input and output formats.

The open source version of Inference Engine includes the following plugins:

| PLUGIN               | DEVICE TYPES |
| ---------------------| -------------|
| CPU plugin           | Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
| GPU plugin           | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
| GNA plugin           | Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel® Pentium® Silver processor J5005, Intel® Celeron® processor J4005, Intel® Core™ i3-8121U processor |
| MYRIAD plugin        | Intel® Movidius™ Neural Compute Stick powered by the Intel® Movidius™ Myriad™ 2, Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X |
| Heterogeneous plugin | Heterogeneous plugin enables computing for inference on one network on several Intel® devices. |

Inference Engine plugin for Intel® FPGA is distributed only in a binary form as a part of [Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/openvino-toolkit).

## Build on Linux Systems

The software was validated on:
- Ubuntu\* 16.04 (64-bit) with default GCC\* 5.4.0
- CentOS\* 7.4 (64-bit) with default GCC\* 4.8.5

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.5 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- Python 2.7 or higher for Inference Engine Python API wrapper
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.04.12237](https://github.com/intel/compute-runtime/releases/tag/19.04.12237).

### Build Steps
1. Clone submodules:
    ```sh
    cd dldt/inference-engine
    git submodule init
    git submodule update --recursive
    ```
2. Install build dependencies using the `install_dependencies.sh` script in the project root folder.
3. By default, the build enables the Inference Engine GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.04.12237](https://github.com/intel/compute-runtime/releases/tag/19.04.12237) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_CLDNN=ON` CMake build option and skip the installation of the Intel® Graphics Compute Runtime for OpenCL™ Driver.
4. Create a build folder:
```sh
  mkdir build && cd build
```
5. Inference Engine uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```

### Additional Build Options

You can use the following additional build options:

- Internal JIT GEMM implementation is used by default.

- To switch to OpenBLAS\* implementation, use the `GEMM=OPENBLAS` option and `BLAS_INCLUDE_DIRS` and `BLAS_LIBRARIES` CMake options to specify path to the OpenBLAS headers and library. For example use the following options on CentOS\*: `-DGEMM=OPENBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`.

- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` CMake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded from the [MKL-DNN repository](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_lnx_2019.0.1.20180928.tgz).

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically downloaded packages but you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed. 

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine) section for details. 

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
   ```sh
   -DPYTHON_EXECUTABLE=`which python3.7` \
   -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
   ```

- To switch off/on the CPU and GPU plugins, use the `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF` respectively. 
  
5. Adding to your project

    For CMake projects, set an environment variable `InferenceEngine_DIR`:

    ```sh
    export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/
    ```

    Then you can find Inference Engine by `find_package`:

    ```cmake
    find_package(InferenceEngine)

    include_directories(${InferenceEngine_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)
    ```

## Build for Raspbian Stretch* OS

> **NOTE**: Only the MYRIAD plugin is supported.

### Hardware Requirements
* Raspberry Pi\* 2 or 3 with Raspbian\* Stretch OS (32-bit). Check that it's CPU supports ARMv7 instruction set (`uname -m` command returns `armv7l`).

  > **NOTE**: Despite the Raspberry Pi\* CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

You can compile the Inference Engine for Raspberry Pi\* in one of the two ways:
* [Native Compilation](#native-compilation), which is the simplest way, but time-consuming
* [Cross Compilation Using Docker](#cross-compilation-using-docker), which is the recommended way

### Native Compilation
Native compilation of the Inference Engine is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi\* 3.

1. Install dependencies:

  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake libusb-1.0-0-dev
  ```

2. Go to the `inference-engine` directory of the cloned `dldt` repository:

  ```bash
  cd dldt/inference-engine
  ```

3. Initialize submodules:

  ```bash
  git submodule init
  git submodule update --recursive
  ```

4. Create a build folder:

  ```bash
  mkdir build && cd build
  ```

5. Build the Inference Engine:

  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_SSE42=OFF \
        -DTHREADING=SEQ \
        -DENABLE_GNA=OFF .. && make -j2
  ```

### Cross Compilation Using Docker

  This compilation was tested on the following configuration:

  * Host: Ubuntu\* 16.04 (64-bit, Intel® Core™ i7-6700K CPU @ 4.00GHz × 8)
  * Target: Raspbian\* Stretch (32-bit, ARMv7, Raspberry Pi\* 3)

1. Install Docker\*:

  ```bash
  sudo apt-get install -y docker.io
  ```

2. Add a current user to `docker` group:

  ```bash
  sudo usermod -a -G docker $USER
  ```

  Log out and log in for this to take effect.

3. Create a directory named `ie_cross_armhf` and add a text file named `Dockerfile`
with the following content:

  ```docker
  FROM debian:stretch

  USER root

  RUN dpkg --add-architecture armhf && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
      build-essential \
      crossbuild-essential-armhf \
      git \
      wget \
      libusb-1.0-0-dev:armhf \
      libgtk-3-dev:armhf \
      libavcodec-dev:armhf \
      libavformat-dev:armhf \
      libswscale-dev:armhf \
      libgstreamer1.0-dev:armhf \
      libgstreamer-plugins-base1.0-dev:armhf \
      libpython3-dev:armhf \
      python3-pip
      
  RUN wget https://www.cmake.org/files/v3.14/cmake-3.14.3.tar.gz && \
      tar xf cmake-3.14.3.tar.gz && \
      (cd cmake-3.14.3 && ./bootstrap --parallel=$(nproc --all) && make --jobs=$(nproc --all) && make install) && \
      rm -rf cmake-3.14.3 cmake-3.14.3.tar.gz

  ```

  It uses the Debian\* Stretch (Debian 9) OS for compilation because it is a base of the Raspbian\* Stretch.

4. Build a Docker\* image:

  ```bash
  docker image build -t ie_cross_armhf ie_cross_armhf
  ```

5. Run Docker\* container with mounted source code folder from host:

  ```bash
  docker run -it -v /absolute/path/to/dldt:/dldt ie_cross_armhf /bin/bash
  ```

6. While in the container:

    1. Go to the `inference-engine` directory of the cloned `dldt` repository:

      ```bash
      cd dldt/inference-engine
      ```

    2. Create a build folder:

      ```bash
      mkdir build && cd build
      ```

    3. Build the Inference Engine:

      ```bash
      cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE="../cmake/arm.toolchain.cmake" \
          -DTHREADS_PTHREAD_ARG="-pthread" \
          -DENABLE_SSE42=OFF \
          -DTHREADING=SEQ \
          -DENABLE_GNA=OFF .. && make --jobs=$(nproc --all)
      ```

7. Press "Ctrl"+"D" to exit from Docker\*. You can find the resulting binaries in the `dldt/inference-engine/bin/armv7l/` directory and the OpenCV* installation in the `dldt/inference-engine/temp`.
   
>**NOTE**: Native applications that link to cross-compiled Inference Engine library require an extra compilation flag `-march=armv7-a`.

### Additional Build Options

You can use the following additional build options:

- Required versions of OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically downloaded packages but you already have installed OpenCV packages configured in your environment, you may need to clean the `OpenCV_DIR` environment variable before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed. 

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine) section for details.

- To build Python API wrapper, install `libpython3-dev:armhf` and `python3-pip` packages using `apt-get`, then install `numpy` and `cython` python modules using `pip3` command and add the following cmake options:
```sh
  -DENABLE_PYTHON=ON \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.5 \
  -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.5m.so \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.5
```

## Build on Windows Systems

The software was validated on:
- Microsoft\* Windows\* 10 (64-bit) with Visual Studio 2017 and Intel® C++ Compiler 2018 Update 3

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.5 or higher
- [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download) and [mingw64\* runtime dependencies](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download).
- [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0 to build the Inference Engine on Windows.
- (Optional) [Intel® Graphics Driver for Windows* [25.20] driver package](https://downloadcenter.intel.com/download/28646/Intel-Graphics-Windows-10-DCH-Drivers?product=80939).
- Python 3.4 or higher for Inference Engine Python API wrapper

### Build Steps
1. Clone submodules:
    ```sh
    git submodule init
    git submodule update --recursive
    ```
2. Download and install [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0
3. Install OpenBLAS:
    1. Download [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download)
    2. Unzip the downloaded package to a directory on your machine. In this document, this directory is referred to as `<OPENBLAS_DIR>`.
4. By default, the build enables the Inference Engine GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [download and install the Intel® Graphics Driver for Windows* [25.20] driver package](https://downloadcenter.intel.com/download/28646/Intel-Graphics-Windows-10-DCH-Drivers?product=80939) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_CLDNN=ON` CMake build option and skip the installation of the Intel® Graphics Driver.    
5. Create build directory:
    ```sh
    mkdir build
    ```
6. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution:
```sh
cd build
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 18.0" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DICCLIB="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\compiler\lib" ..
```

7. Build generated solution in Visual Studio 2017 or run `cmake --build . --config Release` to build from the command line.

8. Before running the samples, add paths to TBB and OpenCV binaries used for the build to the `%PATH%` environment variable. By default, TBB binaries are downloaded by the CMake-based script to the `<dldt_repo>/inference-engine/temp/tbb/lib` folder, OpenCV binaries - to the `<dldt_repo>/inference-engine/temp/opencv_4.1.0/bin` folder.

### Additional Build Options

- Internal JIT GEMM implementation is used by default.
- To switch to OpenBLAS GEMM implementation, use the `-DGEMM=OPENBLAS` CMake option and specify path to OpenBLAS using the `-DBLAS_INCLUDE_DIRS=<OPENBLAS_DIR>\include` and `-DBLAS_LIBRARIES=<OPENBLAS_DIR>\lib\libopenblas.dll.a` options. Prebuilt OpenBLAS\* package can be downloaded [here](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download). mingw64* runtime dependencies can be downloaded [here](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download).
- To switch to the optimized MKL-ML\* GEMM implementation, use the `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` CMake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded from the [MKL-DNN repository](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_win_2019.0.1.20180928.zip).

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically downloaded packages but you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine) section for details.

- To switch off/on the CPU and GPU plugins, use the `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF` respectively.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
   ```sh
   -DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
   -DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
   -DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include"
   ```

### Building Inference Engine with Ninja Build System

```sh
call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\bin\ipsxe-comp-vars.bat" intel64 vs2017
set CXX=icl
set CC=icl
:: clean TBBROOT value set by ipsxe-comp-vars.bat, required TBB package will be downloaded by dldt cmake script
set TBBROOT=
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Build on macOS Systems

> **NOTE**: The current version of the OpenVINO™ toolkit for macOS* supports inference on Intel CPUs only.

The software was validated on:
- macOS\* 10.14, 64-bit

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.5 or higher
- Clang\* compiler from Xcode\* 10.1
- Python\* 3.4 or higher for the Inference Engine Python API wrapper

### Build Steps
1. Clone submodules:
    ```sh
    cd dldt/inference-engine
    git submodule init
    git submodule update --recursive
    ```
2. Install build dependencies using the `install_dependencies.sh` script in the project root folder.
3. Create a build folder:
```sh
  mkdir build
```
4. Inference Engine uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```
### Additional Build Options

You can use the following additional build options:
- Internal JIT GEMM implementation is used by default.
- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` cmake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17.1/mklml_mac_2019.0.1.20180928.tgz)

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically downloaded packages but you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine) section for details.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
```sh
  -DPYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 \
  -DPYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib \
  -DPYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m
```

## Use Custom OpenCV Builds for Inference Engine

> **NOTE**: The recommended and tested version of OpenCV is 4.1. The minimum supported version is 3.4.0.  

Required versions of OpenCV packages are downloaded automatically during the building Inference Engine library. If the build script can not find and download the OpenCV package that is supported on your platform, you can use one of the following options:

* Download the most suitable version from the list of available pre-build packages from [https://download.01.org/opencv/2019/openvinotoolkit](https://download.01.org/opencv/2019/openvinotoolkit) from the `<release_version>/inference_engine` directory.

* Use a system provided OpenCV package (e.g with running the `apt install libopencv-dev` command). The following modules must be enabled: `imgcodecs`, `videoio`, `highgui`.

* Get the OpenCV package using a package manager: pip, conda, conan etc. The package must have the development components included (header files and CMake scripts).

* Build OpenCV from source using the [build instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html) on the OpenCV site. 
  
After you got the built OpenCV library, perform the following preparation steps before running the Inference Engine build:
  
1. Set the `OpenCV_DIR` environment variable to the directory where the `OpenCVConfig.cmake` file of you custom OpenCV build is located.
2. Disable the package automatic downloading with using the `-DENABLE_OPENCV=OFF` option for CMake-based build script for Inference Engine.

## <a name="optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2"></a>(Optional) Additional Installation Steps for the Intel Movidius Neural Compute Stick and Neural Compute Stick 2

> **NOTE**: These steps are only required if you want to perform inference on Intel® Movidius™ Neural Compute Stick or the Intel® Neural Compute Stick 2 using the Inference Engine MYRIAD Plugin. See also [Intel® Neural Compute Stick 2 Get Started](https://software.intel.com/en-us/neural-compute-stick/get-started)

### For Linux, Raspbian\* Stretch OS

1. Add the current Linux user to the `users` group:
```sh
sudo usermod -a -G users "$(whoami)"
```
Log out and log in for it to take effect.

2. To perform inference on Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2, install the USB rules as follows:
```sh
cat <<EOF > 97-myriad-usbboot.rules
SUBSYSTEM=="usb", ATTRS{idProduct}=="2150", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
EOF
```
```sh
sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/
```
```sh
sudo udevadm control --reload-rules
```
```sh
sudo udevadm trigger
```
```sh
sudo ldconfig
```
```sh
rm 97-myriad-usbboot.rules
```

### For Windows

For Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2, install the Movidius™ VSC driver:
1. Go to the `<DLDT_ROOT_DIR>/inference-engine/thirdparty/movidius/MovidiusDriver` directory, where the `DLDT_ROOT_DIR` is the directory to which the DLDT repository was cloned. 
2. Right click on the `Movidius_VSC_Device.inf` file and choose **Install** from the pop up menu.

You have installed the driver for your Intel® Movidius™ Neural Compute Stick or Intel® Neural Compute Stick 2.

## Next Steps

Congratulations, you have built the Inference Engine. To get started with the OpenVINO™ DLDT, proceed to the Get Started guides:

* [Get Started with Deep Learning Deployment Toolkit on Linux*](../get-started-linux.md)

## Additional Resources

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

---
\* Other names and brands may be claimed as the property of others.