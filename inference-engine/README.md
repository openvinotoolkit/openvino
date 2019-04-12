## Repository components

The Inference Engine can infer models in different formats with various input and output formats.

The open source version of Inference Engine includes the following plugins:

| PLUGIN               | DEVICE TYPES |
| ---------------------| -------------|
| CPU plugin           | Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
| GPU plugin           | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
| GNA plugin           | Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel® Pentium® Silver processor J5005, Intel® Celeron® processor J4005, Intel® Core™ i3-8121U processor |
| Heterogeneous plugin | Heterogeneous plugin enables computing for inference on one network on several Intel® devices. |

Inference Engine plugins for Intel® FPGA and Intel® Movidius™ Neural Compute Stick are distributed only in a binary form as a part of [Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/openvino-toolkit).

## Build on Linux\* Systems

The software was validated on:
- Ubuntu\* 16.04 (64-bit) with default GCC\* 5.4.0
- CentOS\* 7.4 (64-bit) with default GCC\* 4.8.5
- [Intel® Graphics Compute Runtime for OpenCL™ Driver package 18.28.11080](https://github.com/intel/compute-runtime/releases/tag/18.28.11080).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- Python 2.7 or higher for Inference Engine Python API wrapper

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
  make -j16
```
You can use the following additional build options:
- Internal JIT GEMM implementation is used by default.
- To switch to OpenBLAS\* implementation, use `GEMM=OPENBLAS` option and `BLAS_INCLUDE_DIRS` and `BLAS_LIBRARIES` cmake options to specify path to OpenBLAS headers and library, for example use the following options on CentOS\*: `-DGEMM=OPENBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`

- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` cmake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_lnx_2019.0.1.20180928.tgz)

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
```sh
  -DPYTHON_EXECUTABLE=`which python3.7` \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
```

- To switch on/off the CPU and GPU plugins, use `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF`.

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

## Build on Windows\* Systems:

The software was validated on:
- Microsoft\* Windows\* 10 (64-bit) with Visual Studio 2017 and Intel® C++ Compiler 2018 Update 3
- [Intel® Graphics Driver for Windows* [24.20] driver package](https://downloadcenter.intel.com/download/27803/Graphics-Intel-Graphics-Driver-for-Windows-10?v=t).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download) and [mingw64\* runtime dependencies](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download).
- [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0 to build the Inference Engine on Windows.
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
4. Create build directory:
    ```sh
    mkdir build
    ```
5. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution:
```sh
cd build
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 18.0" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DICCLIB="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\compiler\lib" ..
```

- Internal JIT GEMM implementation is used by default.
- To switch to OpenBLAS GEMM implementation, use -DGEMM=OPENBLAS cmake option and specify path to OpenBLAS using `-DBLAS_INCLUDE_DIRS=<OPENBLAS_DIR>\include` and `-DBLAS_LIBRARIES=<OPENBLAS_DIR>\lib\libopenblas.dll.a` options. Prebuilt OpenBLAS\* package can be downloaded [here](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download), mingw64* runtime dependencies [here](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download)
- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` cmake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_win_2019.0.1.20180928.zip)

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- Required versions of TBB and OpenCV packages are downloaded automatically by the CMake-based script. If you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
```sh
  -DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
  -DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
  -DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include"
```

6. Build generated solution in Visual Studio 2017 or run `cmake --build . --config Release` to build from the command line.

7. Before running the samples, add paths to TBB and OpenCV binaries used for the build to the %PATH% environment variable. By default, TBB binaries are downloaded by the CMake-based script to the `<dldt_repo>/inference-engine/temp/tbb/lib` folder, OpenCV binaries - to the `<dldt_repo>/inference-engine/temp/opencv_4.1.0/bin` folder.

### Building Inference Engine with Ninja

```sh
call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\bin\ipsxe-comp-vars.bat" intel64 vs2017
set CXX=icl
set CC=icl
:: clean TBBROOT value set by ipsxe-comp-vars.bat, required TBB package will be downloaded by dldt cmake script
set TBBROOT=
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Build on macOS\* Systems

The software was validated on:
- macOS\* 10.14, 64-bit

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
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
  make -j16
```
You can use the following additional build options:
- Internal JIT GEMM implementation is used by default.
- To switch to the optimized MKL-ML\* GEMM implementation, use `-DGEMM=MKL` and `-DMKLROOT=<path_to_MKL>` cmake options to specify a path to unpacked MKL-ML with the `include` and `lib` folders. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17.1/mklml_mac_2019.0.1.20180928.tgz)

- Threading Building Blocks (TBB) is used by default. To build the Inference Engine with OpenMP* threading, set the `-DTHREADING=OMP` option.

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
```sh
  -DPYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 \
  -DPYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib \
  -DPYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m
```


---
\* Other names and brands may be claimed as the property of others.
