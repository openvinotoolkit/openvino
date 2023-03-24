# Build OpenVINO™ Runtime for Linux systems

The software was validated on:
- Ubuntu 18.04 (64-bit) with default GCC 7.5.0
- Ubuntu 20.04 (64-bit) with default GCC 9.3.0
- Red Hat Enterprise Linux 8.2 (64-bit) with default GCC 8.5.0

> **NOTE**: To build on CentOS 7 (64-bit), refer to [Building OpenVINO on CentOS 7 Guide](https://github.com/openvinotoolkit/openvino/wiki/Building-OpenVINO-on-CentOS-7-Guide)

## Software requirements 

- [CMake](https://cmake.org/download/) 3.13 or higher
- GCC 7.5 or higher to build OpenVINO Runtime
- Python 3.7 or higher for OpenVINO Runtime Python API
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441](https://github.com/intel/compute-runtime/releases/tag/19.41.14441) to enable inference on Intel integrated GPUs.

## How to build

1. Clone OpenVINO repository and init submodules:
   ```sh
   git clone https://github.com/openvinotoolkit/openvino.git
   cd openvino
   git submodule update --init --recursive
   ```
   (Optional) For users in China, clone submodules via gitee mirrors
   ```sh
   chmod +x scripts/submodule_update_with_gitee.sh
   ./scripts/submodule_update_with_gitee.sh
   ```

2. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   chmod +x install_build_dependencies.sh
   ```
   ```sh
   sudo ./install_build_dependencies.sh
   ```
   > **NOTE**: By default, the build enables the OpenVINO Runtime GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441](https://github.com/intel/compute-runtime/releases/tag/19.41.14441) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_INTEL_GPU=OFF` CMake build option and skip the installation of the Intel® Graphics Compute Runtime for OpenCL™ Driver.

3. Create a build folder:
```sh
  mkdir build && cd build
```
4. OpenVINO Runtime uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
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
  1. Install all additional packages (e.g., cython and opencv) listed in the `/src/bindings/python/src/compatibility/openvino/requirements-dev.txt` file:
     ```sh
     pip install -r requirements-dev.txt
     ```
  2. Enable the `-DENABLE_PYTHON=ON` option in the CMake step above (Step 4). To specify an exact Python version, use the following options:
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

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

