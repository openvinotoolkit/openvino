# How to build OpenVINO

The guide provides the basic information about the process of building OpenVINO.

```mermaid
gantt 
    %% Use a hack for centry as a persantage
    dateFormat YYYY
    axisFormat %y
    todayMarker off
    title       OpenVINO getting started pipeline
    Setup environment :env, 2000, 1716w
    Build openvino :crit, build, after env, 1716w
    Run tests :active, run, after build, 1716w
```

## Setup environment

This charpter describe how to prepare your machine for OpenVINO development.

### Software requirements 

For the OpenVINO build next tools are required:
<details><summary>Windows</summary>
<p>
    
- [CMake]\* 3.14 or higher
- Microsoft\* Visual Studio 2019, version 16.8 or later
- (Optional) Intel® Graphics Driver for Windows* (30.0) [driver package](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html).
- Python 3.7 or higher for OpenVINO Runtime Python API
- [Git for Windows*](https://gitforwindows.org/)
    
</p>
</details>
<details><summary>Linux</summary>
<p>

- [CMake] 3.13 or higher
- GCC 7.5 or higher to build OpenVINO Runtime
- Python 3.7 or higher for OpenVINO Runtime Python API
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441] to enable inference on Intel integrated GPUs.
    
<p>
</details>
<details><summary>Mac</summary>
<p>

- [brew](https://brew.sh) package manager to install additional dependencies. Use [install brew](https://brew.sh) guide to achieve this.
- [CMake]\* 3.13 or higher
  ```sh
  % brew install cmake
  ```
- Clang\* compiler, git and other command line tools from Xcode\* 10.1 or higher:
  ```sh
  % xcode-select --install
  ``` 
- The step to install python and python libraries is different depending on host architecture:
  - **x86_64** Python\* 3.7 or higher for the OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # let's have a look what python versions are available in brew
  % brew search python
  % # select preferred version of python based on available ones, e.g. 3.10
  % brew install python@3.10
  ```
  - **arm64** Select universal2 installer from [Python releases] download page and install `python-3.X.Y-macos11.pkg` image. This allows to have universal python libraries and build x86_64 OpenVINO Python API and Development tools.

- Additional `pip` dependencies to build OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # update pip and setuptools to newer versions
  % python3 -m pip install -U pip setuptools
  % python3 -m pip install cython
  ```
  In order to build OpenVINO Python API and Development tools as wheel packages, additionally install requirements (after OpenVINO repo clone):
  ```sh
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
  ```
- (native compilation only) libusb library for MYRIAD device and `pkg-config` which is used to find `libusb` files:
  ```sh
  % brew install pkg-config libusb
  ```
- (Optional; native compilation only) Latest version of TBB library. By default, OpenVINO downloads prebuilt version of TBB 2020.4 library, but if you want to use latest (add `-DENABLE_SYSTEM_TBB=ON` additionally to cmake configuration step):
  ```sh
  % brew install tbb
  ```
    
</p>
</details>

### Clone OpenVINO Repository

Clone OpenVINO repository and init submodules:
```sh
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
```


<details><summary>(Optional) For users in China, clone submodules via gitee mirrors</summary>
<p>

```sh
chmod +x scripts/submodule_update_with_gitee.sh
./scripts/submodule_update_with_gitee.sh
```

<p>
</details>

Congratulate! Now you are ready to build the OpenVINO.

## How to build

<details><summary>Windows</summary>
<p>

> **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to download and install the Intel® Graphics Driver for Windows (26.20) [driver package](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Driver.

1. Create build directory:
    ```sh
    mkdir build && cd build
    ```
2. In the `build` directory, run `cmake` to fetch project dependencies and
   generate a Visual Studio solution.

   For Microsoft\* Visual Studio 2019 x64 architecture:
    ```sh
    cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..
    ```

   For Microsoft\* Visual Studio 2019 ARM architecture:
    ```sh
    cmake -G "Visual Studio 16 2019" -A ARM -DCMAKE_BUILD_TYPE=Release ..
    ```

   For Microsoft\* Visual Studio 2019 ARM64 architecture:
    ```sh
    cmake -G "Visual Studio 16 2019" -A ARM64 -DCMAKE_BUILD_TYPE=Release ..
    ```

3. Build generated solution in Visual Studio or run
   `cmake --build . --config Release --verbose -j8` to build from the command line. Note that this process may take some time.

4. Before running the samples, add paths to the Threading Building Blocks (TBB) and OpenCV binaries used for
   the build to the `%PATH%` environment variable. By default, TBB binaries are
   downloaded by the CMake-based script to the `<openvino_repo>/temp/tbb/bin`
   folder, OpenCV binaries to the `<openvino_repo>/temp/opencv_4.5.0/opencv/bin`
   folder.

### Additional Build Options

- Internal JIT GEMM implementation is used by default.

- Threading Building Blocks (TBB) is used by default. To build Inference
  Engine with OpenMP threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by
  the CMake-based script. If you want to use the automatically-downloaded
  packages but you have already installed TBB or OpenCV packages configured in
  your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR`
  environment variables before running the `cmake` command; otherwise they won't
  be downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, refer to the 
  [Use Custom OpenCV Builds](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation#Building-with-custom-OpenCV)
  section for details.

- To switch off/on the CPU and GPU plugins, use the `cmake` options
  `-DENABLE_INTEL_CPU=ON/OFF` and `-DENABLE_INTEL_GPU=ON/OFF` respectively.

- To build the OpenVINO Runtime Python API:
   1. First, install all additional packages (e.g., cython and opencv) listed in the
     `src\bindings\python\src\compatibility\openvino\requirements-dev.txt` file:
      ```sh
      pip install -r requirements-dev.txt
      ```
  2. Second, enable the `-DENABLE_PYTHON=ON` in the CMake (Step #4) option above. To
  specify an exact Python version, use the following options:
     ```sh
     -DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
     -DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
     -DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include"
     ```

- OpenVINO runtime compilation options:
  `-DENABLE_OV_ONNX_FRONTEND=ON` enables the building of the ONNX importer.

### Building OpenVINO with Ninja* Build System

```sh
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

<p>
</details>

<details><summary>Linux</summary>
<p>

1. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   chmod +x install_build_dependencies.sh
   ```
   ```sh
   ./install_build_dependencies.sh
   ```
   > **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441] before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Compute Runtime for OpenCL™ Driver.

2. Create a build folder:
```sh
  mkdir build && cd build
```
3. OpenVINO Runtime uses a CMake-based build system. In the created `build`
   directory, run `cmake` to fetch project dependencies and create Unix
   makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```
The process may take some time to finish.

### Additional Build Options

You can use the following additional build options:

- Threading Building Blocks (TBB) is used by default. To build the OpenVINO 
  Runtime with OpenMP\* threading set the `-DTHREADING=OMP` option.

- For IA32 operation systems, use [ia32.linux.toolchain.cmake](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/ia32.linux.toolchain.cmake) CMake toolchain file:

   ```sh
   cmake -DCMAKE_TOOLCHAIN_FILE=<openvino_repo>/cmake/toolchains/ia32.linux.toolchain.cmake ..
   ```

- Required versions of TBB and OpenCV packages are downloaded automatically by
  the CMake-based script. If you want to use the automatically downloaded
  packages but you have already installed TBB or OpenCV packages configured in
  your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR`
  environment variables before running the `cmake` command, otherwise they
  will not be downloaded and the build may fail if incompatible versions were
  installed.

- If the CMake-based build script can not find and download the OpenCV package
  that is supported on your platform, or if you want to use a custom build of
  the OpenCV library, see how to [Use Custom OpenCV Builds](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation#Building-with-custom-OpenCV).

- To build the OpenVINO Runtime Python API:
  1. Install all additional packages (e.g., cython and opencv) listed in the
     `/src/bindings/python/src/compatibility/openvino/requirements-dev.txt` file:
     ```sh
     pip install -r requirements-dev.txt
     ```
  2. Enable the `-DENABLE_PYTHON=ON` option in the CMake step above (Step 4). To specify an exact Python version, use the following
     options:
     ```
     -DPYTHON_EXECUTABLE=`which python3.7` \
     -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
     -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
     ```
  3. To build a wheel package (.whl), enable the `-DENABLE_WHEEL=ON` option in the CMake step above (Step 4):
  4. After the build process finishes, export the newly built Python libraries to the user environment variables: 
     ```
     export PYTHONPATH=PYTHONPATH:<openvino_repo>/bin/intel64/Release/python_api/python3.7
     export LD_LIBRARY_PATH=LD_LIBRARY_PATH:<openvino_repo>/bin/intel64/Release
     ```
     or install the wheel with pip:
     ```
     pip install <openvino_repo>/build/wheel/openvino-2022.2.0-000-cp37-cp37-manylinux_2_35_x86_64.whl
     ```

- To switch the CPU and GPU plugins off/on, use the `cmake` options
  `-DENABLE_INTEL_CPU=ON/OFF` and `-DENABLE_INTEL_GPU=ON/OFF` respectively.

- OpenVINO runtime compilation options:
  `-DENABLE_OV_ONNX_FRONTEND=ON` enables the building of the ONNX importer.


<p>
</details>

<details><summary>Mac</summary>
<p>

This guide shows how to build OpenVINO Runtime for later inference on Apple with:

<details><summary>Intel CPU</summary>
<p>

This can be done using two ways:
- Compile on Intel CPU host using native compilation. Note, that [Build steps](#build-steps) show this scenario.
- Cross-compile on OSX Apple Silicon.

1. Create a build folder:
```sh
mkdir build && cd build
```
2. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
```
> **Note:** By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. Please, read [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation) for this information.
3. (CMake build) Build OpenVINO project:
```sh
cmake --build . --config Release --jobs=$(nproc --all)
```
All built binaries are located in `<openvino_source_dir>/bin/intel64/Release/` and wheel packages are located in `<openvino_build_dir>/wheels`.

4. (Optional install) Once you have built OpenVINO, you can install artifacts to a preferred location:
```sh
cmake -DCMAKE_INSTALL_PREFIX=<installation location> -P cmake_install.cmake
```

### Cross-compilation 

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allows to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled even on Apple Silicon machines, then such artifacts can be run on both Intel CPU hosts and Apple Silicon hosts (using [Rosetta]). For this, try to compile OpenVINO using the steps above, but adding `-DCMAKE_OSX_ARCHITECTURES=x86_64 -DENABLE_INTEL_MYRIAD=OFF` on cmake configure stage. But, **don't enable any system library usage explicitly** via CMake options, because they have `arm64` architecture, e.g.:
```sh
file /opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib
/opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

If you will see the errors like below:
```sh
ld: warning: ignoring file /opt/homebrew/lib/libopencv_imgproc.4.6.0.dylib, building for macOS-x86_64 but attempting to link with file built for macOS-arm64
Undefined symbols for architecture x86_64:
  "cv::Mat::Mat(cv::Size_<int>, int, void*, unsigned long)", referenced from:
      _image_resize in opencv_c_wrapper.cpp.o
      _image_save in opencv_c_wrapper.cpp.o
....
ld: symbol(s) not found for architecture x86_64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```
Disable its usage in cmake or totally remove such library from the system (e.g. `brew uninstall opencv`), because it's pure arm64 and cannot be used to compile x86_64 binaries.

> **Note:** using such way OpenVINO Intel CPU plugin can be cross-compiled, because MYRIAD plugin cannot be linked against `arm64` version of `libusb`

Or you have to explicitly find / compile x86_64 (or even `universal2`) dependencies by yourself and pass it to OpenVINO cmake scripts. E.g. compile oneTBB using additional option `-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"`, install and then set `export TBBROOT=<universal oneTBB install root>` which will be used by OpenVINO.
    
<p>
</details>

<details><summary>ARM</summary>
<p>

There are two options how to use OpenVINO on Apple Silicon:
- (Native) Compile OpenVINO for arm64 architecture with extra module [OpenVINO ARM plugin](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/arm_plugin) location in [OpenVINO Contrib](https://github.com/openvinotoolkit/openvino_contrib). Note, build steps will cover this as a default scenario.
- (Rosetta) Compile Intel CPU plugin `x86_64` architecture and run under [Rosetta].

1. Clone contrib repo:
```sh
cd ..
git clone https://github.com/openvinotoolkit/openvino_contrib.git
cd openvino_contrib
git submodule update --init --recursive
cd ../openvino
```
2. Create a build folder:
```sh
  mkdir build && cd build
```
3. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -DCMAKE_BUILD_TYPE=Release -DOPENVINO_EXTRA_MODULES=../openvino_contrib/modules/arm_plugin ..
```
> **Note:** By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. Please, read [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation) for this information.
4. (CMake build) Build OpenVINO project:
```sh
cmake --build . --config Release --jobs=$(nproc --all)
```
All built binaries are located in `<openvino_source_dir>/bin/<arm64 | intel64>/Release/` and wheel packages are located in `<openvino_build_dir>/wheels`. 

5. (Optional install) Once you have built OpenVINO, you can install artifacts to a preferred location:
```sh
cmake -DCMAKE_INSTALL_PREFIX=<installation location> -P cmake_install.cmake
```

### Building x86_64 binaries

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allows to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled as x86_64 binary, then run on Apple Silicon hosts using [Rosetta]. For this, first of all Rosetta must be installed:

```sh
softwareupdate --install-rosetta
```

Then try to compile OpenVINO using the steps above, but adding `-DCMAKE_OSX_ARCHITECTURES=x86_64` on cmake configure stage. But, **don't enable any system library usage explicitly** via CMake options, because they have `arm64` architecture, e.g.:
```sh
file /opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib
/opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

The same for other external dependencies like `libusb`. If you want to enable extra functionality like enable MYRIAD plugin build, you need to provide either x86_64 or universal2 `libusb` library. All other steps are the same as for usual compilation: build, install.

> **Note:** since you are building with `universal2` python libraries, wheel package is created with name `openvino-2022.3.0-000-cp39-cp39-macosx_12_0_universal2.whl` and have proper `universal2` tags, so can *potentially* be used on both Apple Silicon and Intel CPU.


<p>
</details>


<p>
</details>

## Run application

[CMake]:https://cmake.org/download/
[Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441]:https://github.com/intel/compute-runtime/releases/tag/19.41.14441
[Rosetta]:https://support.apple.com/en-us/HT211861
