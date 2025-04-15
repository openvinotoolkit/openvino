# Build OpenVINO™ Runtime for macOS systems (Apple Silicon)

This guide shows how to build OpenVINO Runtime for later inference on Apple Silicon on macOS.

There are two options how to use OpenVINO on Apple Silicon:
- (Native) Compile OpenVINO for arm64 architecture. Note, build steps will cover this as a default scenario.
- (Rosetta) Compile Intel CPU plugin `x86_64` architecture and run under [Rosetta](https://support.apple.com/en-us/HT211861).

The software was validated on:
- macOS 11.x, 12.x, 13.x, arm64

## Software requirements

- [brew](https://brew.sh) package manager to install additional dependencies. Use [install brew](https://brew.sh) guide to achieve this.

- Installation step for python and python libraries varies depending on the host architecture:
  - **arm64** Python 3.9 - 3.12 for the OpenVINO Runtime Python API:
  ```sh
  % # let's have a look what python versions are available in brew
  % brew search python
  % # select preferred version of python based on available ones, e.g. 3.11
  % brew install python@3.11
  ```
  - **x86_64** Select universal2 installer from [Python releases](https://www.python.org/downloads/macos/) download page and install `python-3.X.Y-macos11.pkg` image. This allows you to have universal python libraries of OpenVINO Python API (build x86_64).

- Clang compiler and other command line tools from Xcode 10.1 or higher:
  ```sh
  % xcode-select --install
  ```
- [CMake](https://cmake.org/download/) 3.13 or higher and other build dependencies:
  ```sh
  % brew install cmake scons fdupes git-lfs ninja
  ```
- (arm64 only) Product and samples dependencies:
  ```sh
  % brew install tbb pugixml flatbuffers snappy protobuf
  ```
- Additional `pip` dependencies to build OpenVINO Runtime Python API:
  ```sh
  % # update pip and setuptools to newer versions
  % python3 -m pip install -U pip
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/requirements.txt
  ```
  Additional install requirements (after OpenVINO repo clone) in order to build OpenVINO Python API as wheel packages:
  ```sh
  % python3 -m pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
  ```

## How to build

1. (Get sources) Clone the repository and submodules:
```sh
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init
```
2. Create a build folder:
```sh
mkdir build && cd build
```
3. (CMake configure) OpenVINO project uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create build rules:
```sh
cmake -G "Ninja Multi-Config" -DENABLE_SYSTEM_PUGIXML=ON -DENABLE_SYSTEM_SNAPPY=ON -DENABLE_SYSTEM_PROTOBUF=ON ..
```
> **NOTE**: By default OpenVINO CMake scripts try to introspect the system and enable all possible functionality based on that. You can look at the CMake output and see warnings, which show that some functionality is turned off and the corresponding reason, guiding what to do to install additionally to enable unavailable functionality. Additionally, you can change CMake options to enable / disable some functionality, add / remove compilation flags, provide custom version of dependencies like TBB, PugiXML, Protobuf. For more information, see [CMake Options for Custom Compilation](./cmake_options_for_custom_compilation.md).
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

Then try to compile OpenVINO using the steps above, but adding `-DCMAKE_OSX_ARCHITECTURES=x86_64` on cmake configure stage. But, **don't enable any system library usage explicitly** via CMake options (remove all `-DENABLE_SYSTEM_*` options), because they have `arm64` architecture, e.g.:
```sh
file /opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib
/opt/homebrew/Cellar/tbb/2021.5.0_2/lib/libtbb.12.5.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

The same goes for other external dependencies like `pugixml`.

> **NOTE**: When building with `universal2` python libraries, wheel package is created with the `openvino-2022.3.0-000-cp311-cp311-macosx_12_0_universal2.whl` name and have a proper `universal2` tags, so they can *potentially* be used on both Apple Silicon and Intel CPU.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)