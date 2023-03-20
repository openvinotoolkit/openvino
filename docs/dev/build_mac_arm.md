# Build OpenVINO™ Runtime for macOS systems (Apple Silicon)

This guide shows how to build OpenVINO Runtime for later inference on Apple Silicon & Intel MYRIAD devices on OSX. 

There are two options how to use OpenVINO on Apple Silicon:
- (Native) Compile OpenVINO for arm64 architecture with extra module [OpenVINO ARM plugin](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/arm_plugin) location in [OpenVINO Contrib](https://github.com/openvinotoolkit/openvino_contrib). Note, build steps will cover this as a default scenario.
- (Rosetta) Compile Intel CPU plugin `x86_64` architecture and run under [Rosetta](https://support.apple.com/en-us/HT211861).

The software was validated on:
- macOS 11.x, 12.x, arm64

## Software requirements 

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

- [CMake](https://cmake.org/download/) 3.13 or higher:
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

## How to build

1. (Get sources) Clone submodules:
```sh
git clone https://github.com/openvinotoolkit/openvino.git
git clone https://github.com/openvinotoolkit/openvino_contrib.git
cd openvino_contrib
git submodule update --init
cd ../openvino
git submodule update --init
```
2. Create a build folder:
```sh
mkdir build && cd build
```
3. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -DCMAKE_BUILD_TYPE=Release -DOPENVINO_EXTRA_MODULES=../openvino_contrib/modules/arm_plugin ..
```
> **NOTE**: By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, OpenCV, Protobuf. For more information, see [CMake Options for Custom Compilation](./cmake_options_for_custom_comiplation.md).
4. (CMake build) Build OpenVINO project:
```sh
cmake --build . --config Release --parallel $(sysctl -n hw.ncpu)
```
All built binaries are located in `<openvino_source_dir>/bin/<arm64 | intel64>/Release/` and wheel packages are located in `<openvino_build_dir>/wheels`. 

5. (Optional install) Once you have built OpenVINO, you can install artifacts to a preferred location:
```sh
cmake -DCMAKE_INSTALL_PREFIX=<installation location> -P cmake_install.cmake
```

### Building x86_64 binaries

Since OSX version 11.x and Xcode version 12.2, the Apple development tools allow to compile arm64 code on x86 hosts and vice-versa. Based on this, OpenVINO can be compiled as x86_64 binary, then run on Apple Silicon hosts using [Rosetta](https://support.apple.com/en-us/HT211861). To do this, you must first install Rosetta:

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

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

 