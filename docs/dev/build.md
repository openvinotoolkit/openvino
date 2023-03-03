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

> **NOTE**: For the details on how to build static OpenVINO, refer to [Building static OpenVINO libraries](static_libaries.md)

<details><summary>Windows</summary>
<p>
    
- [CMake] 3.13 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
  > **NOTE**: Native Microsoft Visual Studio for WoA is available since 2022. 
- Python 3.7 or higher for OpenVINO Runtime Python API
  > **NOTE**: Python for ARM64 is available since [3.11](https://www.python.org/downloads/windows/) version. 
- [Git for Windows*]
- (Windows on ARM only) [LLVM for Windows on ARM (WoA)](https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/LLVM-15.0.6-woa64.exe)
  > **NOTE**: After installation, make sure `clang-cl` compiler is available from `PATH`. 
    
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
- [CMake] 3.13 or higher
  ```sh
  % brew install cmake
  ```
- Clang compiler, git and other command line tools from Xcode 10.1 or higher:
  ```sh
  % xcode-select --install
  % brew install git-lfs
  ``` 
- Installation step for python and python libraries varies depending on the host architecture:
  - **x86_64** Python 3.7 or higher for the OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # let's have a look what python versions are available in brew
  % brew search python
  % # select preferred version of python based on available ones, e.g. 3.11
  % brew install python@3.11
  ```
  - **arm64** Select universal2 installer from [Python releases](https://www.python.org/downloads/macos/) download page and install `python-3.X.Y-macos11.pkg` image. This allows to have universal python libraries, build x86_64 OpenVINO Python API and Development tools.

- Additional `pip` dependencies to build OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # update pip and setuptools to newer versions
  % python3 -m pip install -U pip setuptools
  % python3 -m pip install cython
  ```
  Additional install requirements (after OpenVINO repo clone) in order to build OpenVINO Python API and Development tools as wheel packages:
  ```sh
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
  ```
- (Optional; native compilation only) Latest version of TBB library. By default, OpenVINO downloads prebuilt version of TBB 2020.4 library, if you want to use latest (add `-DENABLE_SYSTEM_TBB=ON` additionally to cmake configuration step):
  ```sh
  % brew install tbb
  ```
    
</p>
</details>
<details><summary>Mac (ARM)</summary>
<p>

- [brew](https://brew.sh) package manager to install additional dependencies. Use [install brew](https://brew.sh) guide to achieve this.

- Installation step for python and python libraries varies depending on the host architecture:
  - **arm64** Python 3.7 or higher for the OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # let's have a look what python versions are available in brew
  % brew search python
  % # select preferred version of python based on available ones, e.g. 3.11
  % brew install python@3.11
  ```
  - **x86_64** Select universal2 installer from [Python releases] download page and install `python-3.X.Y-macos11.pkg` image. This allows to have universal python libraries, build x86_64 OpenVINO Python API and Development tools.

- [CMake] 3.13 or higher:
  ```sh
  % brew install cmake
  ```
- Clang compiler, git and other command line tools from Xcode 10.1 or higher:
  ```sh
  % xcode-select --install
  % brew install git-lfs
  ```
- (arm64 only) `scons` to build ARM compute library:
  ```sh
  % python3 -m pip install scons
  ```
- (arm64 only) TBB library for threading:
  ```sh
  % brew install tbb
  ```
- Additional `pip` dependencies to build OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # update pip and setuptools to newer versions
  % python3 -m pip install -U pip setuptools
  % python3 -m pip install cython
  ```
  Additional install requirements (after OpenVINO repo clone) in order to build OpenVINO Python API and Development tools as wheel packages:
  ```sh
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
  ```
    
</p>
</details>
<details><summary>Android</summary>
<p>

- [CMake] 3.13 or higher
- Android NDK (this guide has been validated with r20 release)

</p>
</details>
<details><summary>WebAssembly</summary>
<p>

- [Docker Engine](https://docs.docker.com/engine/install/)

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
<details><summary>(Optional) To build OpenVINO extra modules</summary>
<p>

    ```sh
    git clone https://github.com/openvinotoolkit/openvino_contrib.git
    cd openvino_contrib
    git submodule update --init --recursive
    ```

<p>
</details>

Congratulations! You are ready to build the OpenVINO.

## How to build

<details><summary>Windows</summary>
<p>

OpenVINO can be compiled for different architectures on Windows: X64 or ARM64. In order to build for ARM64 architecture, the machine with Windows on ARM is required as only native compilation is supported (see [similar documents](https://www.linaro.org/blog/how-to-set-up-windows-on-arm-for-llvm-development/#:~:text=Install%20the%20Latest%20LLVM%20for,PATH%20yourself%2C%20as%20described%20above.) for details).

Supported configurations:
- Windows 10 x86 64-bit or higher with Visual Studio 2019 or higher build for X64 architecture.
- Windows on ARM (shortly WoA) to build for ARM64 architecture. OpenVINO was validated on [Windows DevKit 2023](https://developer.qualcomm.com/hardware/windows-on-snapdragon/windows-dev-kit-2023)

> **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to download and install the Intel® Graphics Driver for Windows (26.20) [driver package](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Driver.

1. Create build directory:
    ```sh
    mkdir build && cd build
    ```
2. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution.

   On Windows x86 64-bits:
    ```sh
    cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release <openvino>
    ```

   On Windows on ARM for ARM64 architecture:
    ```sh
    cmake -G "Visual Studio 16 2019" -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules/arm_plugin -DCMAKE_BUILD_TYPE=Release <openvino>
    ```

3. Build generated solution in Visual Studio or run `cmake --build . --config Release --verbose -j8` to build from the command line. Be aware that this process may take some time.

4. Before running the samples, add paths to the Threading Building Blocks (TBB) binaries used for the build to the `%PATH%` environment variable. By default, TBB binaries are downloaded by the CMake-based script to the `<openvino>/temp/tbb/bin` folder.

### Additional Build Options

- Internal JIT GEMM implementation is used by default.

- Threading Building Blocks (TBB) is used by default. To build Inference Engine with OpenMP threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically-downloaded packages but you have already installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command; otherwise they won'tnbe downloaded and the build may fail if incompatible versions were installed.

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](./cmake_options_for_custom_comiplation.md#Building-with-custom-OpenCV) section for details.

- To build the OpenVINO Runtime Python API:
  1. First, install all additional packages (e.g., cython and opencv) listed in the file:
      ```sh
      pip install -r <openvino>\src\bindings\python\src\compatibility\openvino\requirements-dev.txt
      ```
  2. Second, enable the `-DENABLE_PYTHON=ON` in the CMake (Step #4) option above. To specify an exact Python version, use the following options:
     ```sh
     -DPYTHON_EXECUTABLE="C:\Program Files\Python11\python.exe" ^
     -DPYTHON_LIBRARY="C:\Program Files\Python11\libs\python11.lib" ^
     -DPYTHON_INCLUDE_DIR="C:\Program Files\Python11\include"
     ```
  3. To build a wheel package (.whl), enable the `-DENABLE_WHEEL=ON` option in the CMake step above (Step 4):
  4. After the build process finishes, export the newly built Python libraries to the user environment variables:
     ```
     set PYTHONPATH=<openvino_repo>/bin/<arch>/Release/python_api/python3.11;%PYTHONPATH%
     set OPENVINO_LIB_PATH=<openvino_repo>/bin/<arch>/Release;%OPENVINO_LIB_PATH%
     ```
     or install the wheel with pip:
     ```
     pip install build/wheel/openvino-2023.0.0-9612-cp11-cp11-win_arm64.whl
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

The software was validated on:
- Ubuntu 18.04 (64-bit) with default GCC 7.5.0
- Ubuntu 20.04 (64-bit) with default GCC 9.3.0
- Red Hat Enterprise Linux 8.2 (64-bit) with default GCC 8.5.0

> **NOTE**: To build on CentOS 7 (64-bit), refer to [Building OpenVINO on CentOS 7 Guide](https://github.com/openvinotoolkit/openvino/wiki/Building-OpenVINO-on-CentOS-7-Guide)

1. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   chmod +x install_build_dependencies.sh
   ```
   ```sh
   sudo ./install_build_dependencies.sh
   ```
   > **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441] before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Compute Runtime for OpenCL™ Driver.

2. Create a build folder:
```sh
  mkdir build && cd build
```
3. OpenVINO Runtime uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```
The process may take some time to finish.

### Additional Build Options

You can use the following additional build options:

- For IA32 operation systems, use [ia32.linux.toolchain.cmake](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/ia32.linux.toolchain.cmake) CMake toolchain file:

   ```sh
   cmake -DCMAKE_TOOLCHAIN_FILE=<openvino_repo>/cmake/toolchains/ia32.linux.toolchain.cmake ..
   ```

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

The software was validated on:
- macOS 10.x, 11.x, 12.x x86 64-bit
- macOS 11.x, 12.x, arm64 (cross-compilation)

1. Create a build folder:
```sh
mkdir build && cd build
```
2. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
```
> **NOTE**: By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. For more information, see [CMake Options for Custom Compilation](./cmake_options_for_custom_comiplation.md).
3. (CMake build) Build OpenVINO project:
```sh
cmake --build . --config Release --parallel $(sysctl -n hw.ncpu)
```
All built binaries are located in `<openvino_source_dir>/bin/intel64/Release/` and wheel packages are located in `<openvino_build_dir>/wheels`.

4. (Optional install) Once you have built OpenVINO, you can install artifacts to a preferred location:
```sh
cmake -DCMAKE_INSTALL_PREFIX=<installation location> -P cmake_install.cmake
```

### Cross-compilation 

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allows to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled even on Apple Silicon machines, then such artifacts can be run on both Intel CPU hosts and Apple Silicon hosts (using [Rosetta]). To do this, add `-DCMAKE_OSX_ARCHITECTURES=x86_64 -DENABLE_INTEL_MYRIAD=OFF` in the cmake configuration step when compiling OpenVINO following the steps above. **Don't enable any system library usage explicitly** via CMake options, because they have `arm64` architecture, e.g.:
```sh
file /opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib
/opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

If you will see the errors like the one below:
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
Disable its usage in cmake or completely remove such library from the system (e.g. `brew uninstall opencv`), because it's pure arm64 and cannot be used to compile x86_64 binaries.

> **NOTE**: By using such way, the OpenVINO Intel CPU plugin can be cross-compiled, because MYRIAD plugin cannot be linked against `arm64` version of `libusb`

Or you have to explicitly find / compile x86_64 (or even `universal2`) dependencies by yourself and pass it to OpenVINO cmake scripts. E.g. compile oneTBB using additional option `-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"`, install and then set `export TBBROOT=<universal oneTBB install root>` which will be used by OpenVINO.
    
<p>
</details>

<details><summary>ARM</summary>
<p>

There are two options how to use OpenVINO on Apple Silicon:
- (Native) Compile OpenVINO for arm64 architecture with extra module [OpenVINO ARM plugin](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/arm_plugin) location in [OpenVINO Contrib](https://github.com/openvinotoolkit/openvino_contrib). Note, build steps will cover this as a default scenario.
- (Rosetta) Compile Intel CPU plugin `x86_64` architecture and run under [Rosetta].

The software was validated on:
- macOS 11.x, 12.x, arm64

> **NOTE**: Before you proceed, make sure that openvino and openvino_contrib repositories were cloned.

1. Create a build folder:
```sh
mkdir build && cd build
```
2. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -DCMAKE_BUILD_TYPE=Release -DOPENVINO_EXTRA_MODULES=../openvino_contrib/modules/arm_plugin ..
```
> **NOTE**: By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. For more information, see [CMake Options for Custom Compilation](./cmake_options_for_custom_comiplation.md).
3. (CMake build) Build OpenVINO project:
```sh
cmake --build . --config Release --parallel $(sysctl -n hw.ncpu)
```
All built binaries are located in `<openvino_source_dir>/bin/<arm64 | intel64>/Release/` and wheel packages are located in `<openvino_build_dir>/wheels`. 

4. (Optional install) Once you have built OpenVINO, you can install artifacts to a preferred location:
```sh
cmake -DCMAKE_INSTALL_PREFIX=<installation location> -P cmake_install.cmake
```

### Building x86_64 binaries

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allow to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled as x86_64 binary, then run on Apple Silicon hosts using [Rosetta]. To do this, you must first install Rosetta:

```sh
softwareupdate --install-rosetta
```

Then try to compile OpenVINO using the steps above, but adding `-DCMAKE_OSX_ARCHITECTURES=x86_64` on cmake configure stage. But, **don't enable any system library usage explicitly** via CMake options, because they have `arm64` architecture, e.g.:
```sh
file /opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib
/opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

The same goes for other external dependencies like `libusb`. If you want to enable extra functionality like enable MYRIAD plugin build, you need to provide either x86_64 or universal2 `libusb` library. All other steps are the same as for usual compilation: build, install.

> **NOTE**: When building with `universal2` python libraries, wheel package is created with the `openvino-2022.3.0-000-cp39-cp39-macosx_12_0_universal2.whl` name and have a proper `universal2` tags, so they can *potentially* be used on both Apple Silicon and Intel CPU.


<p>
</details>


<p>
</details>

<details><summary>Android</summary>
<p>

1. Download and unpack [Android NDK](https://developer.android.com/ndk/downloads). Let's assume that `~/Downloads` is used as a working folder.
  ```sh
  cd ~/Downloads
  wget https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip

  unzip android-ndk-r20-linux-x86_64.zip
  mv android-ndk-r20 android-ndk
  ```

2. Create a build folder:
  ```sh
  mkdir build
  ```

3. Change working directory to `build` and run `cmake` to create makefiles. Then run `make`.
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

  * `ANDROID_ABI` specifies target architecture:
    * `x86_64` for x64 build
    * `armeabi-v7a with NEON` for ARM with NEON support
    * `arm64-v8a` for ARM 64 bits
  * `ANDROID_PLATFORM` - Android API version
  * `ANDROID_STL` specifies that shared C++ runtime is used. Copy `~/Downloads/android-ndk/sources/cxx-stl/llvm-libc++/libs/x86_64/libc++_shared.so` from Android NDK along with built binaries

4. To reduce the binaries size, use `strip` tool from NDK:

```bash
~/Downloads/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/x86_64-linux-android/bin/strip openvino/bin/intel64/Release/lib/*.so
```

<p>
</details>

<details><summary>Raspbian Stretch</summary>
<p>

### Hardware Requirements
* Raspberry Pi 2 or 3 with Raspbian Stretch OS (32 or 64-bit).

  > **NOTE**: Despite the Raspberry Pi CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

You can compile the OpenVINO Runtime for Raspberry Pi in one of the two ways:
* [Native Compilation](#native-compilation), which is the simplest way, but time-consuming
* [Cross Compilation Using Docker](#cross-compilation-using-docker), which is the recommended way

### Native Compilation
Native compilation of the OpenVINO Runtime is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi 3.

1. Install dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake scons build-essential
  ```
2. Clone the repositories:
```
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino.git 
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino_contrib.git 
```
3. Go to the cloned `openvino` repository:

  ```bash
  cd openvino/
  ```
4. Create a build folder:

  ```bash
  mkdir build && cd build/
  ```
5. Build the OpenVINO Runtime:
* for MYRIAD support only:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DOPENVINO_EXTRA_MODULES=<OPENVINO_CONTRIB_PATH>/openvino_contrib/modules/arm_plugin \
        -DARM_COMPUTE_SCONS_JOBS=$(nproc --all) \
  .. && cmake --build . --parallel 
  ```

### Cross Compilation Using Docker*

To cross-compile ARM CPU plugins using pre-configured `Dockerfile` you can use the following instruction: [Build OpenCV, OpenVINO™ and the plugin using pre-configured Dockerfile](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin#approach-1-build-opencv-openvino-and-the-plugin-using-pre-configured-dockerfile-cross-compiling-the-preferred-way).

### Additional Build Options

- To build Python API, install `libpython3-dev:armhf` and `python3-pip`
  packages using `apt-get`; then install `numpy` and `cython` python modules
  via `pip3`, adding the following options:
   ```sh
   -DENABLE_PYTHON=ON \
   -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
   -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
   ```

</p>
</details>

<details><summary>WebAssembly</summary>
<p>

1. Run docker image and mount a volume with OpenVINO source code:
```sh
$ docker pull emscripten/emsdk
$ docker run -it --rm -v `pwd`:/openvino emscripten/emsdk
```
2. (CMake configure) Run cmake configure step using helper emscripten command:
```sh
$ mkdir build && cd build
$ emcmake cmake -DCMAKE_BUILD_TYPE=Release /openvino
```
3. (CMake build) Build OpenVINO project:
```sh
$ emmake make -j$(nproc)
```
`openvino.wasm` and `openvino.js` files are located in:
- `<openvino_source_dir>/bin/ia32/Release/` on host machine file system.
- `/openvino/bin/ia32/Release` in docker environment.
These files can be used in browser applications. 

</p>
</details>

<details><summary>Docker Image</summary>
<p>

For details on how to build Intel® Distribution of OpenVINO™ toolkit in a Docker image, follow this [guide](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles/ubuntu18/build_custom).

</p>
</details>

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)

[CMake]:https://cmake.org/download/
[Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441]:https://github.com/intel/compute-runtime/releases/tag/19.41.14441
[Rosetta]:https://support.apple.com/en-us/HT211861
