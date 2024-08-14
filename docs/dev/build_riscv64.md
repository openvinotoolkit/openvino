# Cross compile OpenVINO™ Runtime for RISCV64 systems
This guide shows how to build OpenVINO Runtime for 64-bit RISC-V devices. Due to limited resources, cross compilation is used now for building OpenVINO targeting RISC-V development boards.

Cross compilation was tested on the following hosts:
- Ubuntu 22.04 (64-bit), x64

The software was validated on the following devices:
- [Lichee Pi 4A](https://wiki.sipeed.com/hardware/en/lichee/th1520/lp4a.html) with RVV 0.7.1
- [Banana Pi BPI-F3](https://www.banana-pi.org/en/banana-pi-sbcs/175.html) with RVV 1.0


## Software requirements

- [CMake](https://cmake.org/download/) 3.13 or higher
- GCC 7.5 or higher (for non-RVV) / [xuantie-gnu-toolchain](https://github.com/XUANTIE-RV/xuantie-gnu-toolchain) (for RVV)
- Python 3.10 for OpenVINO Runtime Python API

## How to build
### Build with RVV
0. Prerequisite - build `xuantie-gnu-toolchain` and `qemu`:
   ```sh
   git clone https://github.com/XUANTIE-RV/xuantie-gnu-toolchain.git
   cd xuantie-gnu-toolchain
   ./configure --prefix=<xuantie_install_path>
   make linux build-qemu -j$(nproc)
   ```
   > **NOTE**: The `build-qemu` target is optional, as it is used to build the `qemu` simulator. However, it is recommended to build the `qemu` simulator, since it is much more convenient to validate the software on your host than on your devices.

1. Clone OpenVINO repository and init submodules:
   ```sh
   git clone https://github.com/openvinotoolkit/openvino.git
   cd openvino
   git submodule update --init --recursive
   ```

2. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   sudo ./install_build_dependencies.sh
   ```

3. Create a build folder:
   ```sh
   mkdir build && cd build
   ``` 

4. To cross compile OpenVINO Runtime for RISC-V devices, run `cmake` with specified `CMAKE_TOOLCHAIN_FILE` and `RISCV_TOOLCHAIN_ROOT`:
   ```sh
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=<openvino_install_path> \
     -DENABLE_INTEL_CPU=ON \
     -DENABLE_INTEL_GPU=OFF \
     -DENABLE_INTEL_NPU=OFF \
     -DENABLE_AUTO=ON \
     -DENABLE_HETERO=ON \
     -DENABLE_MULTI=ON \
     -DENABLE_JS=OFF \
     -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/<toolchain_file> \
     -DRISCV_TOOLCHAIN_ROOT=<xuantie_install_path> \
     -DENABLE_TESTS=ON \
     -DENABLE_SAMPLES=ON
   ```
   > **NOTE**: To build OpenVINO Runtime for different versions of RVV, you just need to specify corresponding toolchain files. For exmaple, you can replace `<toolchain_file>` with `riscv64-071-thead-gnu.toolchain.cmake` for RVV 0.7.1 and `riscv64-100-thead-gnu.toolchain.cmake` for RVV 1.0 respectively.

   Then run `make` to build the project:
   ```sh
   make install -j$(nproc)
   ```

### Build without RVV
0. Prerequisite - build `riscv-gnu-toolchain`:
   ```sh
   git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
   cd riscv-gnu-toolchain
   ./configure --prefix=/opt/riscv
   make linux build-qemu -j$(nproc)
   ```
   > **NOTE**: The `build-qemu` target is optional here as well. More information can be seen [here](https://github.com/riscv-collab/riscv-gnu-toolchain).

1. Clone OpenVINO repository and init submodules:
   ```sh
   git clone https://github.com/openvinotoolkit/openvino.git
   cd openvino
   git submodule update --init --recursive
   ```

2. Install build dependencies using the `install_build_dependencies.sh` script in the
   project root folder.
   ```sh
   sudo ./install_build_dependencies.sh
   ```

3. Create a build folder:
   ```sh
   mkdir build && cd build
   ``` 

4. To cross compile OpenVINO Runtime for RISC-V devices, run `cmake` with specified `CMAKE_TOOLCHAIN_FILE` and `RISCV_TOOLCHAIN_ROOT`:
   ```sh
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=<openvino_install_path> \
     -DENABLE_INTEL_CPU=ON \
     -DENABLE_INTEL_GPU=OFF \
     -DENABLE_INTEL_NPU=OFF \
     -DENABLE_AUTO=ON \
     -DENABLE_HETERO=ON \
     -DENABLE_MULTI=ON \
     -DENABLE_JS=OFF \
     -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64-071-thead-gnu.toolchain.cmake \
     -DRISCV_TOOLCHAIN_ROOT=/opt/riscv \
     -DENABLE_TESTS=ON \
     -DENABLE_SAMPLES=ON
   ```
   > **NOTE**: The `riscv-gnu-toolchain` is build as there are essential files used for cross compilation under `/opt/riscv/sysroot`. The latest stable versions of Clang or GCC both support compiling source code into RISC-V instructions, so it is acceptable to choose your preferable compilers by specifying `-DCMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`. But remember to add the key `-DCMAKE_SYSROOT=/opt/riscv/sysroot`, otherwise many fundamental headers and libs could not be found during cross compilation. 

   Then run `make` to build the project:
   ```sh
   make install -j$(nproc)
   ```

### Build the OpenVINO Runtime Python API

To enable cross-compilation with python, the library `libpython3-dev:riscv64` should be on the host machine.

When installing packages using the utilities `apt` or `apt-get` the packages are downloaded from apt software repositories. On Ubuntu the apt software repositories are defined in the `/etc/apt/sources.list` file or in separate files under the `/etc/apt/sources.list.d/` directory. Host machine contains host-specific repositories (for example, x86-x64) in these files. 

1. Add riscv64 repositories to download riscv64-specific packages:
    ```sh
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy main >> riscv64-sources.list
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy universe >> riscv64-sources.list
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main >> riscv64-sources.list
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> riscv64-sources.list
    mv riscv64-sources.list /etc/apt/sources.list.d/
    dpkg --add-architecture riscv64
    apt-get update -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/riscv64-sources.list
    ```

2. Install `libpython3-dev:riscv64` using `apt-get`:
    ```sh
    apt-get install -y --no-install-recommends libpython3-dev:riscv64
    ```
   Create  symbolink to allow python to find `riscv64-linux-gnu/python3.10/pyconfig.h` in `/usr/include/python3.10/` (this header is initially stored in `/usr/include/riscv64-linux-gnu/`)
    ```sh
    ln -s /usr/include/riscv64-linux-gnu/ /usr/include/python3.10/
    ```

3. Add the keys `-DENABLE_PYTHON=ON -DENABLE_PYTHON_PACKAGING=ON -DENABLE_WHEEL=ON` to cmake command during OpenVINO build.

> **Note**: Currently only Python 3.10 on Ubuntu 22.04 is verified. So the target device must have Python 3.10 in this case.

### How to use OpenVINO builds

- On host machine:

  If you want to quickly verify the correctness of the executable files, you can launch the executable files by `qemu` on your host machine:
  ```sh
  <xuantie_install_path>/bin/qemu-riscv64 -cpu=<target_cpu> <executable_file_path>
  ```
  For example, to launch tests on Lichee Pi 4A:
  ```sh
  cd <openvino_repo>/bin/riscv64/Release
  <xuantie_install_path>/bin/qemu-riscv64 -cpu c910v ./ov_cpu_func_tests
  ```
  Or to launch tests on devices with RVV 1.0:
  ```sh
  cd <openvino_repo>/bin/riscv64/Release
  <xuantie_install_path>/bin/qemu-riscv64 -cpu c908v ./ov_cpu_func_tests
  ```

- On target devices:
  
  - After the building process, there will be the directory `bin` under `<openvino_repo>`. You can just copy this folder to the target device and launch needed executable files (such as `ov_cpu_func_tests` or `benchmark_app`)

  - If you want to write you own C++ application or Python script which will use C++/Python API of OpenVINO (for example, simple python script to infer models), you need to build OpenVINO with `-DCMAKE_INSTALL_PREFIX=<openvino_install_path>`. The folder `<openvino_install_path>` is lighter than `bin` and contains only needed files for the development of other applications with OpenVINO support. You also need to copy this folder to the target machine and activate environment using `source <openvino_install_path>/setupvars.sh`.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

