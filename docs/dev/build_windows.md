# Build OpenVINO™ Runtime for Windows systems

OpenVINO can be compiled for different architectures on Windows: X64 or ARM64. In order to build for ARM64 architecture, the machine with Windows on ARM is required as only native compilation is supported (see [similar documents](https://www.linaro.org/blog/how-to-set-up-windows-on-arm-for-llvm-development/#:~:text=Install%20the%20Latest%20LLVM%20for,PATH%20yourself%2C%20as%20described%20above.) for details).

Supported configurations:
- Windows 10 x86 64-bit or higher with Visual Studio 2019 or higher build for X64 architecture.
- Windows on ARM (shortly WoA) to build for ARM64 architecture. OpenVINO was validated on [Windows DevKit 2023](https://developer.qualcomm.com/hardware/windows-on-snapdragon/windows-dev-kit-2023)

## Software requirements 

- [CMake](https://cmake.org/download/) 3.13 or higher
- Microsoft Visual Studio 2019 or higher, version 16.3 or later
  > **NOTE**: Native Microsoft Visual Studio for WoA is available since 2022. 
- Python 3.7 or higher for OpenVINO Runtime Python API
  > **NOTE**: Python for ARM64 is available since [3.11](https://www.python.org/downloads/windows/) version. 
- [Git for Windows*]
- (Windows on ARM only) [LLVM for Windows on ARM (WoA)](https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/LLVM-15.0.6-woa64.exe)
  > **NOTE**: After installation, make sure `clang-cl` compiler is available from `PATH`. 

## How to build

> **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to download and install the Intel® Graphics Driver for Windows (26.20) [driver package](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Driver.

1. Clone submodules:
    ```sh
    git clone https://github.com/openvinotoolkit/openvino.git
    cd openvino
    git submodule update --init --recursive
    ```
   (Extra for WoA) To build on Windows on ARM with ARM plugin:
    ```sh
    git clone https://github.com/openvinotoolkit/openvino_contrib.git
    cd openvino_contrib
    git submodule update --init --recursive
    ```

2. Create build directory:
    ```sh
    mkdir build && cd build
    ```
3. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution.

   On Windows x86 64-bits:
    ```sh
    cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release <openvino>
    ```

   On Windows on ARM for ARM64 architecture:
    ```sh
    cmake -G "Visual Studio 16 2019" -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules/arm_plugin -DCMAKE_BUILD_TYPE=Release <openvino>
    ```

4. Build generated solution in Visual Studio or run `cmake --build . --config Release --verbose -j8` to build from the command line. Be aware that this process may take some time.

5. Before running the samples, add paths to the Threading Building Blocks (TBB) binaries used for the build to the `%PATH%` environment variable. By default, TBB binaries are downloaded by the CMake-based script to the `<openvino>/temp/tbb/bin` folder.

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

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

