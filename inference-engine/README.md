## Build on Linux\* Systems

The software was validated on:
- Ubuntu\* 16.04 with default GCC\* 5.4.0
- CentOS\* 7.4 with default GCC\* 4.8.5 (using clDNN library built separately with GCC\* 5.2)
- [Intel® Graphics Compute Runtime for OpenCL™ Driver package 18.28.11080](https://github.com/intel/compute-runtime/releases/tag/18.28.11080).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- GCC\* 5.2 or higher to build the Compute Library for Deep Neural Networks (clDNN library)
- OpenBLAS\*

### Build Steps
1. Install OpenBLAS and other dependencies using the `install_dependencies.sh` script in the project root folder.
2. Create a build folder:
```sh
  mkdir build
```
3. Inference Engine uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j16
```
You can use the following additional build options:
- Use `BLAS_INCLUDE_DIRS` and `BLAS_LIBRARIES` cmake options to specify path to OpenBLAS headers and library, for example use the following options on CentOS\*: `-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`
- To build clDNN from sources, please specify the `-DENABLE_CLDNN_BUILD=ON` option for `cmake`. By default pre-built version of the clDNN library is used. 
- To switch on/off the CPU and GPU plugins, use `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF`.

## Build on Windows\* Systems:

The software was validated on:
- Microsoft\* Windows\* 10 with Visual Studio 2017 and Intel® C++ Compiler 2018 Update 3
- [Intel® Graphics Driver for Windows* [24.20] driver package](https://downloadcenter.intel.com/download/27803/Graphics-Intel-Graphics-Driver-for-Windows-10?v=t).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download) or 
- [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0 to build the Inference Engine on Windows.

### Build Steps
1. Download and install [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0
2. Install OpenBLAS:
    1. Download [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download)
    2. Unzip the downloaded package to a directory on your machine. In this document, this directory is referred to as `<OPENBLAS_DIR>`.
3. Create build directory:
    ```sh
    mkdir build
    ```
4. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution:
```sh
cd build
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 18.0" -DOS_FOLDER=ON ^
    -DBLAS_INCLUDE_DIRS=<OPENBLAS_DIR>\include ^
    -DBLAS_LIBRARIES=<OPENBLAS_DIR>\lib\libopenblas.dll.a ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DICCLIB="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\compiler\lib" ..
```

5. Build generated solution in Visual Studio 2017 or run `cmake --build .` to build from the command line.

---
\* Other names and brands may be claimed as the property of others.
