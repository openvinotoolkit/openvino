# Build OpenVINO™ Runtime for Linux systems

The software was validated on:
- Ubuntu 18.04 (64-bit) with default GCC 7.5.0
- Ubuntu 20.04 (64-bit) with default GCC 9.3.0
- Red Hat Enterprise Linux 8.2 (64-bit) with default GCC 8.5.0

> **NOTE**: To build on CentOS 7 (64-bit), refer to [Building OpenVINO on CentOS 7 Guide](https://github.com/openvinotoolkit/openvino/wiki/Building-OpenVINO-on-CentOS-7-Guide)

## Software requirements 

- [CMake](https://cmake.org/download/) 3.13 or higher
- GCC 7.5 or higher to build OpenVINO Runtime
- Python 3.8 - 3.11 for OpenVINO Runtime Python API
- (Optional) Install Intel® Graphics Compute Runtime for OpenCL™ Driver package to enable inference on Intel integrated GPUs. Select a driver package from the table below depending on what version of Ubuntu you are installing on.

  | Ubuntu | Driver package |
  | --- | ----------- |
  | 22.04 | [23.13.26032.30](https://github.com/intel/compute-runtime/releases/tag/23.13.26032.30) |
  | 20.04 | [22.24.23453](https://github.com/intel/compute-runtime/releases/tag/22.24.23453) |
  | 18.04 | [21.38.21026](https://github.com/intel/compute-runtime/releases/tag/21.38.21026) |

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

- OpenVINO offers several CMake options that can be used to build a custom OpenVINO runtime, which can speed up compilation. These options allow you to skip compilation of other plugins and frontends by disabling/enabling them. You can find a list of available options [here](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md)

- To build the OpenVINO Runtime Python API:
  1. Install all additional packages (e.g., cython and opencv) listed in the `/src/bindings/python/src/compatibility/openvino/requirements-dev.txt` file:
     ```sh
     pip install -r requirements-dev.txt
     ```
  2. Enable the `-DENABLE_PYTHON=ON` option in the CMake step above (Step 4). To specify an exact Python version, use the following options (requires cmake 3.16 and higher):
     ```
     -DPython3_EXECUTABLE=/usr/bin/python3.8
     ```
  3. To build a wheel package (.whl), enable the `-DENABLE_WHEEL=ON` option in the CMake step above (Step 4), and install requirements:
     ```sh
     pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
     ```
  4. After the build process finishes, export the newly built Python libraries to the user environment variables: 
     ```
     export PYTHONPATH=<openvino_repo>/bin/intel64/Release/python:$PYTHONPATH
     export LD_LIBRARY_PATH=<openvino_repo>/bin/intel64/Release:$LD_LIBRARY_PATH
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

