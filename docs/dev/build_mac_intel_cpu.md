# Build OpenVINO™ Runtime for macOS systems (Intel CPU)

This guide shows how to build OpenVINO Runtime for later inference on Intel CPU on macOS with Intel CPU underneath. This can be done using two ways:
- Compile on Intel CPU host using native compilation. Note, that [Build steps](#build-steps) show this scenario.
- Cross-compile on OSX Apple Silicon.

The software was validated on:
- macOS 10.x, 11.x, 12.x, 13.x, x86 64-bit
- macOS 11.x, 12.x, 13.x, arm64 (cross-compilation)

## Software Requirements

- [brew](https://brew.sh) package manager to install additional dependencies. Use [install brew](https://brew.sh) guide to achieve this.
- Installation step for python and python libraries varies depending on the host architecture:
  - **x86_64** Python 3.9 - 3.12 for the OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # let's have a look what python versions are available in brew
  % brew search python
  % # select preferred version of python based on available ones, e.g. 3.11
  % brew install python@3.11
  ```
  - **arm64** Select universal2 installer from [Python releases](https://www.python.org/downloads/macos/) download page and install `python-3.X.Y-macos11.pkg` image. This allows to have universal python libraries, build x86_64 OpenVINO Python API and Development tools.
- [CMake](https://cmake.org/download/) 3.13 or higher and other development tools:
  ```sh
  % brew install cmake scons fdupes git-lfs ninja
  ```
- Clang compiler and other command line tools from Xcode 10.1 or higher:
  ```sh
  % xcode-select --install
  ```
- (Optional; native compilation only, x86_64) Product and samples dependencies:
  ```sh
  % brew install tbb pugixml flatbuffers snappy protobuf
  ```
- Additional `pip` dependencies to build OpenVINO Runtime Python API, Development tools (Model Optimizer, POT and others):
  ```sh
  % # update pip and setuptools to newer versions
  % python3 -m pip install -U pip
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/requirements.txt
  ```
  Additional install requirements (after OpenVINO repo clone) in order to build OpenVINO Python API and Development tools as wheel packages:
  ```sh
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
  ```

## How to build

1. Create a build folder:
```sh
mkdir build && cd build
```
2. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -G "Ninja Multi-Config" -DENABLE_SYSTEM_PUGIXML=ON -DENABLE_SYSTEM_SNAPPY=ON -DENABLE_SYSTEM_PROTOBUF=ON ..
```
> **NOTE**: By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. For more information, see [CMake Options for Custom Compilation](./cmake_options_for_custom_compilation.md).
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

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allows to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled even on Apple Silicon machines, then such artifacts can be run on both Intel CPU hosts and Apple Silicon hosts (using [Rosetta](https://support.apple.com/en-us/HT211861)). To do this, add `-DCMAKE_OSX_ARCHITECTURES=x86_64` in the cmake configuration step when compiling OpenVINO following the steps above. **Don't enable any system library usage explicitly** via CMake options (remove all `-DENABLE_SYSTEM_*` options), because they have `arm64` architecture, e.g.:
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

Or you have to explicitly find / compile x86_64 (or even `universal2`) dependencies by yourself and pass it to OpenVINO cmake scripts. E.g. compile oneTBB using additional option `-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"`, install and then set `export TBBROOT=<universal oneTBB install root>` which will be used by OpenVINO.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)
