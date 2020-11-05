# Build OpenVINO™ Inference Engine

## Contents

- [Introduction](#introduction)
- [Build on Linux\* Systems](#build-on-linux-systems)
  - [Software Requirements](#software-requirements)
  - [Build Steps](#build-steps)
  - [Additional Build Options](#additional-build-options)
- [Build for Raspbian* Stretch OS](#build-for-raspbian-stretch-os)
  - [Hardware Requirements](#hardware-requirements)
  - [Native Compilation](#native-compilation)
  - [Cross Compilation Using Docker\*](#cross-compilation-using-docker)
  - [Additional Build Options](#additional-build-options-1)
- [Build on Windows* Systems](#build-on-windows-systems)
  - [Software Requirements](#software-requirements-1)
  - [Build Steps](#build-steps-1)
  - [Additional Build Options](#additional-build-options-2)
  - [Building Inference Engine with Ninja* Build System](#building-inference-engine-with-ninja-build-system)
- [Build on macOS\* Systems](#build-on-macos-systems)
  - [Software Requirements](#software-requirements-2)
  - [Build Steps](#build-steps-2)
  - [Additional Build Options](#additional-build-options-3)
- [Build on Android\* Systems](#build-on-android-systems)
  - [Software Requirements](#software-requirements-3)
  - [Build Steps](#build-steps-3)
- [Use Custom OpenCV Builds for Inference Engine](#use-custom-opencv-builds-for-inference-engine)
- [Add Inference Engine to Your Project](#add-inference-engine-to-your-project)
- [(Optional) Additional Installation Steps for the Intel® Neural Compute Stick 2](#optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2)
  - [For Linux, Raspbian Stretch* OS](#for-linux-raspbian-stretch-os)
- [Next Steps](#next-steps)
- [Additional Resources](#additional-resources)

## Introduction

The Inference Engine can infer models in different formats with various input
and output formats.

The open source version of Inference Engine includes the following plugins:

| PLUGIN               | DEVICE TYPES |
| ---------------------| -------------|
| CPU plugin           | Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
| GPU plugin           | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
| GNA plugin           | Intel® Speech Enabling Developer Kit, Amazon Alexa\* Premium Far-Field Developer Kit, Intel® Pentium® Silver processor J5005, Intel® Celeron® processor J4005, Intel® Core™ i3-8121U processor |
| MYRIAD plugin        | Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X |
| Heterogeneous plugin | Heterogeneous plugin enables computing for inference on one network on several Intel® devices. |

## Build on Linux\* Systems

The software was validated on:
- Ubuntu\* 18.04 (64-bit) with default GCC\* 7.5.0
- Ubuntu\* 20.04 (64-bit) with default GCC\* 9.3.0
- CentOS\* 7.6 (64-bit) with default GCC\* 4.8.5

### Software Requirements
- [CMake]\* 3.13 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- Python 3.6 or higher for Inference Engine Python API wrapper
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441].
> **NOTE**: Building samples and demos from the Intel® Distribution of OpenVINO™ toolkit package requires CMake\* 3.10 or higher.

### Build Steps
1. Clone submodules:
    ```sh
    cd openvino
    git submodule update --init --recursive
    ```
2. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   chmod +x install_build_dependencies.sh
   ```
   ```sh
   ./install_build_dependencies.sh
   ```
3. By default, the build enables the Inference Engine GPU plugin to infer models
   on your Intel® Processor Graphics. This requires you to
   [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441]
   before running the build. If you don't want to use the GPU plugin, use the
   `-DENABLE_CLDNN=OFF` CMake build option and skip the installation of the
   Intel® Graphics Compute Runtime for OpenCL™ Driver.
4. Create a build folder:
```sh
  mkdir build && cd build
```
5. Inference Engine uses a CMake-based build system. In the created `build`
   directory, run `cmake` to fetch project dependencies and create Unix
   makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```

### Additional Build Options

You can use the following additional build options:

- The default build uses an internal JIT GEMM implementation.

- To switch to an OpenBLAS\* implementation, use the `GEMM=OPENBLAS` option with
  `BLAS_INCLUDE_DIRS` and `BLAS_LIBRARIES` CMake options to specify a path to the
  OpenBLAS headers and library. For example, the following options on CentOS\*:
  `-DGEMM=OPENBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`.

- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL`
  and `-DMKLROOT=<path_to_MKL>` CMake options to specify a path to unpacked
  MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded
  from the Intel® [MKL-DNN repository].

- Threading Building Blocks (TBB) is used by default. To build the Inference
  Engine with OpenMP\* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by
  the CMake-based script. If you want to use the automatically downloaded
  packages but you already have installed TBB or OpenCV packages configured in
  your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR`
  environment variables before running the `cmake` command, otherwise they
  will not be downloaded and the build may fail if incompatible versions were
  installed.

- If the CMake-based build script can not find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, refer to the
  [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine)
  section for details.

- To build the Python API wrapper:
  1. Install all additional packages listed in the
     `/inference-engine/ie_bridges/python/requirements.txt` file:
     ```sh
     pip install -r requirements.txt
     ```
  2. Use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following
     options:
     ```
     -DPYTHON_EXECUTABLE=`which python3.7` \
     -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
     -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
     ```

- To switch the CPU and GPU plugins off/on, use the `cmake` options
  `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF` respectively.

- nGraph-specific compilation options:
  `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` enables the building of the nGraph ONNX importer.
  `-DNGRAPH_DEBUG_ENABLE=ON` enables additional debug prints.

## Build for Raspbian Stretch* OS

> **NOTE**: Only the MYRIAD plugin is supported.

### Hardware Requirements
* Raspberry Pi\* 2 or 3 with Raspbian\* Stretch OS (32-bit). Check that it's CPU supports ARMv7 instruction set (`uname -m` command returns `armv7l`).

  > **NOTE**: Despite the Raspberry Pi\* CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

You can compile the Inference Engine for Raspberry Pi\* in one of the two ways:
* [Native Compilation](#native-compilation), which is the simplest way, but time-consuming
* [Cross Compilation Using Docker*](#cross-compilation-using-docker), which is the recommended way

### Native Compilation
Native compilation of the Inference Engine is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi\* 3.

1. Install dependencies:

  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake libusb-1.0-0-dev
  ```

2. Go to the cloned `openvino` repository:

  ```bash
  cd openvino
  ```

3. Initialize submodules:

  ```bash
  git submodule update --init --recursive
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
        -DENABLE_GNA=OFF .. && make
  ```

### Cross Compilation Using Docker*

  This compilation was tested on the following configuration:

  * Host: Ubuntu\* 18.04 (64-bit, Intel® Core™ i7-6700K CPU @ 4.00GHz × 8)
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
      python3-pip \
      python-minimal \
      python-argparse

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
  docker run -it -v /absolute/path/to/openvino:/openvino ie_cross_armhf /bin/bash
  ```

6. While in the container:

    1. Go to the cloned `openvino` repository:

      ```bash
      cd openvino
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

7. Press **Ctrl+D** to exit from Docker. You can find the resulting binaries
   in the `openvino/bin/armv7l/` directory and the OpenCV*
   installation in the `openvino/inference-engine/temp`.

>**NOTE**: Native applications that link to cross-compiled Inference Engine
library require an extra compilation flag `-march=armv7-a`.

### Additional Build Options

You can use the following additional build options:

- Required versions of OpenCV packages are downloaded automatically by the
  CMake-based script. If you want to use the automatically downloaded packages
  but you already have installed OpenCV packages configured in your environment,
  you may need to clean the `OpenCV_DIR` environment variable before running
  the `cmake` command; otherwise they won't be downloaded and the build may
  fail if incompatible versions were installed.

- If the CMake-based build script cannot find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, see: [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine)
  for details.

- To build Python API wrapper, install `libpython3-dev:armhf` and `python3-pip`
  packages using `apt-get`; then install `numpy` and `cython` python modules
  via `pip3`, adding the following options:
   ```sh
   -DENABLE_PYTHON=ON \
   -DPYTHON_EXECUTABLE=/usr/bin/python3.5 \
   -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.5m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.5
   ```

- nGraph-specific compilation options:
  `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` enables the building of the nGraph ONNX importer.
  `-DNGRAPH_DEBUG_ENABLE=ON` enables additional debug prints.

## Build on Windows* Systems

The software was validated on:
- Microsoft\* Windows\* 10 (64-bit) with Visual Studio 2019

### Software Requirements
- [CMake]\*3.13 or higher
- Microsoft\* Visual Studio 2017, 2019
- (Optional) Intel® Graphics Driver for Windows* (26.20) [driver package].
- Python 3.6 or higher for Inference Engine Python API wrapper
> **NOTE**: Building samples and demos from the Intel® Distribution of OpenVINO™ toolkit package requires CMake\* 3.10 or higher.

### Build Steps

1. Clone submodules:
    ```sh
    git submodule update --init --recursive
    ```
2. By default, the build enables the Inference Engine GPU plugin to infer models
   on your Intel® Processor Graphics. This requires you to [download and install
   the Intel® Graphics Driver for Windows (26.20) [driver package] before
   running the build. If you don't want to use the GPU plugin, use the
   `-DENABLE_CLDNN=OFF` CMake build option and skip the installation of the
   Intel® Graphics Driver.
3. Create build directory:
    ```sh
    mkdir build
    ```
4. In the `build` directory, run `cmake` to fetch project dependencies and
   generate a Visual Studio solution.

   For Microsoft\* Visual Studio 2017:
```sh
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release ..
```

   For Microsoft\* Visual Studio 2019:
```sh
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..
```

5. Build generated solution in Visual Studio or run
   `cmake --build . --config Release` to build from the command line.

6. Before running the samples, add paths to the TBB and OpenCV binaries used for
   the build to the `%PATH%` environment variable. By default, TBB binaries are
   downloaded by the CMake-based script to the `<openvino_repo>/inference-engine/temp/tbb/bin`
   folder, OpenCV binaries to the `<openvino_repo>/inference-engine/temp/opencv_4.5.0/opencv/bin`
   folder.

### Additional Build Options

- Internal JIT GEMM implementation is used by default.

- To switch to OpenBLAS GEMM implementation, use the `-DGEMM=OPENBLAS` CMake
  option and specify path to OpenBLAS using the `-DBLAS_INCLUDE_DIRS=<OPENBLAS_DIR>\include`
  and `-DBLAS_LIBRARIES=<OPENBLAS_DIR>\lib\libopenblas.dll.a` options. Download
  a prebuilt OpenBLAS\* package via the [OpenBLAS] link. mingw64* runtime
  dependencies can be downloaded via the [mingw64\* runtime dependencies] link.

- To switch to the optimized MKL-ML\* GEMM implementation, use the
  `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` CMake options to specify a path to
  unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be
  downloaded from the Intel&reg; [MKL-DNN repository for Windows].

- Threading Building Blocks (TBB) is used by default. To build the Inference
  Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by
  the CMake-based script. If you want to use the automatically-downloaded
  packages but you already have installed TBB or OpenCV packages configured in
  your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR`
  environment variables before running the `cmake` command; otherwise they won't
  be downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine)
  section for details.

- To switch off/on the CPU and GPU plugins, use the `cmake` options
  `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF` respectively.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To
  specify an exact Python version, use the following options:
   ```sh
   -DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
   -DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
   -DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include"
   ```

- nGraph-specific compilation options:
  `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` enables the building of the nGraph ONNX importer.
  `-DNGRAPH_DEBUG_ENABLE=ON` enables additional debug prints.

### Building Inference Engine with Ninja* Build System

```sh
call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\bin\ipsxe-comp-vars.bat" intel64 vs2017
set CXX=icl
set CC=icl
:: clean TBBROOT value set by ipsxe-comp-vars.bat, required TBB package will be downloaded by openvino cmake script
set TBBROOT=
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Build on macOS* Systems

> **NOTE**: The current version of the OpenVINO™ toolkit for macOS* supports
inference on Intel CPUs only.

The software was validated on:
- macOS\* 10.15, 64-bit

### Software Requirements

- [CMake]\* 3.13 or higher
- Clang\* compiler from Xcode\* 10.1 or higher
- Python\* 3.6 or higher for the Inference Engine Python API wrapper
> **NOTE**: Building samples and demos from the Intel® Distribution of OpenVINO™ toolkit package requires CMake\* 3.10 or higher.

### Build Steps

1. Clone submodules:
    ```sh
    cd openvino
    git submodule update --init --recursive
    ```
2. Create a build folder:
```sh
  mkdir build && cd build
```
3. Inference Engine uses a CMake-based build system. In the created `build`
   directory, run `cmake` to fetch project dependencies and create Unix makefiles,
   then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```
### Additional Build Options

You can use the following additional build options:

- Internal JIT GEMM implementation is used by default.

- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and
  `-DMKLROOT=<path_to_MKL>` cmake options to specify a path to unpacked MKL-ML
  with the `include` and `lib` folders. MKL-ML\* [package for Mac] can be downloaded
  [here](https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_mac_2019.0.5.20190502.tgz)

- Threading Building Blocks (TBB) is used by default. To build the Inference
  Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by
  the CMake-based script. If you want to use the automatically downloaded
  packages but you already have installed TBB or OpenCV packages configured in
  your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR`
  environment variables before running the `cmake` command, otherwise they won't
  be downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, refer to the
  [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine)
  section for details.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To
  specify an exact Python version, use the following options:
   - If you installed Python through Homebrew*, set the following flags:
   ```sh
   -DPYTHON_EXECUTABLE=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/bin/python3.7m \
   -DPYTHON_LIBRARY=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib \
   -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m
   ```
   - If you installed Python another way, you can use the following commands to find where the `dylib` and `include_dir` are located, respectively:
   ```sh
   find /usr/ -name 'libpython*m.dylib'
   find /usr/ -type d -name python3.7m
   ```
- nGraph-specific compilation options:
  `-DNGRAPH_ONNX_IMPORT_ENABLE=ON` enables the building of the nGraph ONNX importer.
  `-DNGRAPH_DEBUG_ENABLE=ON` enables additional debug prints.

## Build on Android* Systems

This section describes how to build Inference Engine for Android x86 (64-bit) operating systems.

### Software Requirements

- [CMake]\* 3.13 or higher
- Android NDK (this guide has been validated with r20 release)
> **NOTE**: Building samples and demos from the Intel® Distribution of OpenVINO™ toolkit package requires CMake\* 3.10 or higher.

### Build Steps

1. Download and unpack Android NDK: https://developer.android.com/ndk/downloads. Let's assume that `~/Downloads` is used as a working folder.
  ```sh
  cd ~/Downloads
  wget https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip

  unzip android-ndk-r20-linux-x86_64.zip
  mv android-ndk-r20 android-ndk
  ```

2. Clone submodules
  ```sh
  cd openvino
  git submodule update --init --recursive
  ```

3. Create a build folder:
  ```sh
    mkdir build
  ```

4. Change working directory to `build` and run `cmake` to create makefiles. Then run `make`.
  ```sh
  cd build

  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=~/Downloads/android-ndk/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM=21 \
    -DANDROID_STL=c++_shared \
    -DENABLE_OPENCV=OFF

  make --jobs=$(nproc --all)
  ```

  * `ANDROID_ABI` specifies target architecture (`x86_64`)
  * `ANDROID_PLATFORM` - Android API version
  * `ANDROID_STL` specifies that shared C++ runtime is used. Copy `~/Downloads/android-ndk/sources/cxx-stl/llvm-libc++/libs/x86_64/libc++_shared.so` from Android NDK along with built binaries


## Use Custom OpenCV Builds for Inference Engine

> **NOTE**: The recommended and tested version of OpenCV is 4.4.0.

Required versions of OpenCV packages are downloaded automatically during the
building Inference Engine library. If the build script can not find and download
the OpenCV package that is supported on your platform, you can use one of the
following options:

* Download the most suitable version from the list of available pre-build
  packages from [https://download.01.org/opencv/2020/openvinotoolkit] from the
  `<release_version>/inference_engine` directory.

* Use a system-provided OpenCV package (e.g with running the
  `apt install libopencv-dev` command). The following modules must be enabled:
  `imgcodecs`, `videoio`, `highgui`.

* Get the OpenCV package using a package manager: pip, conda, conan etc. The
  package must have the development components included (header files and CMake
  scripts).

* Build OpenCV from source using the [build instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html) on the OpenCV site.

After you got the built OpenCV library, perform the following preparation steps
before running the Inference Engine build:

1. Set the `OpenCV_DIR` environment variable to the directory where the
   `OpenCVConfig.cmake` file of you custom OpenCV build is located.
2. Disable the package automatic downloading with using the `-DENABLE_OPENCV=OFF`
   option for CMake-based build script for Inference Engine.

## Add Inference Engine to Your Project

For CMake projects, set the `InferenceEngine_DIR` environment variable:

```sh
export InferenceEngine_DIR=/path/to/openvino/build/
```

Then you can find Inference Engine by `find_package`:

```cmake
find_package(InferenceEngine)
include_directories(${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)
```

## (Optional) Additional Installation Steps for the Intel® Neural Compute Stick 2

> **NOTE**: These steps are only required if you want to perform inference on the
Intel® Neural Compute Stick 2 using the Inference Engine MYRIAD Plugin. See also
[Intel® Neural Compute Stick 2 Get Started].

### For Linux, Raspbian\* Stretch OS

1. Add the current Linux user to the `users` group; you will need to log out and
   log in for it to take effect:
```sh
sudo usermod -a -G users "$(whoami)"
```

2. To perform inference on Intel® Neural Compute Stick 2, install the USB rules
as follows:
```sh
cat <<EOF > 97-myriad-usbboot.rules
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

## Next Steps

Congratulations, you have built the Inference Engine. To get started with the
OpenVINO™, proceed to the Get Started guides:

* [Get Started with Deep Learning Deployment Toolkit on Linux*](get-started-linux.md)

## Notice

To enable some additional nGraph features and use your custom nGraph library with the OpenVINO™ binary package,
make sure the following:
- nGraph library was built with the same version which is used in the Inference Engine.
- nGraph library and the Inference Engine were built with the same compilers. Otherwise you might face application binary interface (ABI) problems.

To prepare your custom nGraph library for distribution, which includes collecting all headers, copy
binaries, and so on, use the `install` CMake target.
This target collects all dependencies, prepares the nGraph package and copies it to a separate directory.

## Additional Resources

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

---
\* Other names and brands may be claimed as the property of others.


[Intel® Distribution of OpenVINO™]:https://software.intel.com/en-us/openvino-toolkit
[CMake]:https://cmake.org/download/
[Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441]:https://github.com/intel/compute-runtime/releases/tag/19.41.14441
[MKL-DNN repository]:https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_lnx_2019.0.5.20190502.tgz
[MKL-DNN repository for Windows]:(https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_win_2019.0.5.20190502.zip)
[OpenBLAS]:https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download
[mingw64\* runtime dependencies]:https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download
[https://download.01.org/opencv/2020/openvinotoolkit]:https://download.01.org/opencv/2020/openvinotoolkit
[build instructions]:https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html
[driver package]:https://downloadcenter.intel.com/download/29335/Intel-Graphics-Windows-10-DCH-Drivers
[Intel® Neural Compute Stick 2 Get Started]:https://software.intel.com/en-us/neural-compute-stick/get-started
[OpenBLAS]:https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download
