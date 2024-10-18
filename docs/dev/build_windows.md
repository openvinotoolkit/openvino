# Build OpenVINO™ Runtime for Windows systems

OpenVINO can be compiled for different architectures on Windows: X64 or ARM64. In order to build for ARM64 architecture, the machine with Windows on ARM is required as only native compilation is supported. Refer to [the page](https://learn.arm.com/install-guides/llvm-woa/) for details.

Supported configurations:
- Windows 10 x86 64-bit or higher with Visual Studio 2019 or higher build for X64 architecture.
- Windows on ARM (shortly WoA) to build for ARM64 architecture. OpenVINO was validated on [Windows DevKit 2023](https://developer.qualcomm.com/hardware/windows-on-snapdragon/windows-dev-kit-2023)

## Software requirements

- [CMake](https://cmake.org/download/) 3.13 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
  > **NOTE**: Native Microsoft Visual Studio for WoA has been available since version 3.11.
- Python 3.9 - 3.12 for OpenVINO Runtime Python API
  > **NOTE**: Python for ARM64 is available since [3.11](https://www.python.org/downloads/windows/) version.
- [Git for Windows*]
- (Windows on ARM only) [LLVM for Windows on ARM (WoA)](https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/LLVM-15.0.6-woa64.exe)
  > **NOTE**: After installation, make sure `clang-cl` compiler is available from `PATH`.

## How to build

> **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to download and install the [Intel® Graphics Driver for Windows](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Driver.

1. Clone submodules:
    ```sh
    git clone https://github.com/openvinotoolkit/openvino.git
    cd openvino
    git submodule update --init --recursive
    ```

2. Create build directory:
    ```sh
    mkdir build && cd build
    ```
3. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution:

    ```sh
    cmake -G "Visual Studio 17 2022" <path/to/openvino>
    ```

   > **HINT**: **Generating PDB Files and Debugging Your Build** <br>
   > If you intend to generate PDB files and debug your build, it is essential to set the CMake build type appropriately.
   > You should utilize one of the following CMake build type options: <br>
   >* `-DCMAKE_BUILD_TYPE=RelWithDebInfo`: This option generates PDB files with release information, making it suitable for debugging optimized builds. <br>
   >* `-DCMAKE_BUILD_TYPE=Debug`: This option generates PDB files optimized for debugging, providing comprehensive debugging information.

4. Build generated solution in Visual Studio or run `cmake --build . --config Release --verbose -j<number_of_jobs>` to build from the command line. View the number of available processing units with `WMIC cpu get numberofLogicalProcessors`. Be aware that this process may take some time.

5. Before running the samples, add paths to the Threading Building Blocks (TBB) binaries used for the build to the `%PATH%` environment variable. By default, TBB binaries are downloaded by the CMake-based script to the `<path/to/openvino>/temp/tbb/bin` folder.

### Additional Build Options

- To build the OpenVINO Runtime Python API:
  1. Enable the `-DENABLE_PYTHON=ON` in the CMake (Step #3) option above. To specify an exact Python version, use the following options (requires cmake 3.16 and higher):
     ```sh
     -DPython3_EXECUTABLE="C:\Program Files\Python11\python.exe"
     ```
  2. To build a wheel package (.whl), enable the `-DENABLE_WHEEL=ON` option in the CMake step above (Step 4), and install requirements:
     ```sh
     pip install -r <openvino source tree>\src\bindings\python\wheel\requirements-dev.txt
     ```
  3. After the build process finishes, export the newly built Python libraries to the user environment variables:
     ```
     set PYTHONPATH=<openvino_repo>/bin/<arch>/Release/python;<openvino_repo>/tools/ovc;%PYTHONPATH%
     set OPENVINO_LIB_PATHS=<openvino_repo>/bin/<arch>/Release;<openvino_repo>/temp/tbb/bin
     set PATH=<openvino_repo>/tools/ovc/openvino/tools/ovc:%PATH%
     ```
     or install the wheel with pip:
     ```
     pip install build/wheel/openvino-2023.0.0-9612-cp11-cp11-win_arm64.whl
     ```

### Building OpenVINO with Ninja* Build System

```sh
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

