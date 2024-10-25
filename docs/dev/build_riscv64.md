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
Currently, there are three ways to build OpenVINO Runtime for 64-bit RISC-V platforms:

1. **Recommended**. The build with vectorized (using RVV instructions) primitives for limited scope of operations from [`SHL`](https://github.com/XUANTIE-RV/csi-nn2) using [`xuantie-gnu-toolchain`](https://github.com/XUANTIE-RV/). This GNU Compiler Toolchain supports RVV 0.7.1, ratified RVV 1.0 and Xuantie-specific instruction sets. The vector intrinsics don't use the common prefix `__riscv_`. This method provides the best performance available at the moment.
2. The build without optimized primitives using [`riscv-gnu-toolchain`](https://github.com/riscv-collab/riscv-gnu-toolchain.git). This GNU Compiler Toolchain supports RVV 0.7.1 and ratified RVV 1.0. The vector intrinsics use the common prefix `__riscv_`. However, as  mentioned earlier, this build method doesn't yet provide optimized primitives implemented using the RVV intrinsics.
3. The build without optimized primitives using installed Linux packages. The compilers in these packages don't support RVV intrinsics.

### Steps

0. Prerequisite:
- For target with vectorized primitives from `SHL` - build `xuantie-gnu-toolchain` and `qemu`:
   ```sh
   git clone https://github.com/XUANTIE-RV/xuantie-gnu-toolchain.git
   cd xuantie-gnu-toolchain
   ./configure --prefix=<xuantie_install_path>
   make linux build-qemu -j$(nproc)
   ```
- For target without optimized primitives using `riscv-gnu-toolchain`:
   ```sh
   git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
   cd riscv-gnu-toolchain
   ./configure --prefix=/opt/riscv
   make linux build-qemu -j$(nproc)
   ```
   > **NOTE**: The `build-qemu` target is optional, as it is used to build the `qemu` simulator. However, it is recommended to build the `qemu` simulator, since it is much more convenient to validate the software on your host than on your devices. More information can be seen [here](https://github.com/riscv-collab/riscv-gnu-toolchain).
- For target without optimized primitives using installed Linux packages:
   ```sh
   apt-get update
   apt-get install -y  gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu
   ```

1. Clone OpenVINO repository and init submodules:
   ```sh
   git clone --recursive https://github.com/openvinotoolkit/openvino.git
   cd openvino
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

4. To cross compile OpenVINO Runtime for RISC-V devices, run `cmake` with specified `CMAKE_TOOLCHAIN_FILE` and `RISCV_TOOLCHAIN_ROOT` (the last one is needed only for build using GNU toolchain).
- For target with vectorized primitives from `SHL`:
   ```sh
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=<openvino_install_path> \
     -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/<toolchain_file> \
     -DRISCV_TOOLCHAIN_ROOT=<xuantie_install_path>
   ```
   > **NOTE**: To build OpenVINO Runtime for different versions of RVV, you just need to specify corresponding toolchain files. For example, you can replace `<toolchain_file>` with `riscv64-071-xuantie-gnu.toolchain.cmake` for RVV 0.7.1 and `riscv64-100-xuantie-gnu.toolchain.cmake` for RVV 1.0 respectively.
- For target without optimized primitives using `riscv-gnu-toolchain`:
   ```sh
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=<openvino_install_path> \
     -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64-gnu.toolchain.cmake \
     -DRISCV_TOOLCHAIN_ROOT=/opt/riscv
   ```
   > **NOTE**: The `riscv-gnu-toolchain` is build as there are essential files used for cross compilation under `/opt/riscv/sysroot`. The latest stable versions of Clang or GCC both support compiling source code into RISC-V instructions, so it is acceptable to choose your preferable compilers by specifying `-DCMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`. But remember to add the key `-DCMAKE_SYSROOT=/opt/riscv/sysroot`, otherwise many fundamental headers and libs could not be found during cross compilation. 
- For target without optimized primitives using installed Linux packages:
   ```sh
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=<openvino_install_path> \
     -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64.linux.toolchain.cmake
   ```
   > **NOTE**: By default OpenVINO is built with OpenMP support on RISC-V devices.

   Then run `make` to build the project:
   ```sh
   make install -j$(nproc)
   ```

### (Optional) Build the OpenVINO Runtime Python API
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

3. Add the keys `-DENABLE_PYTHON=ON -DENABLE_WHEEL=ON` to cmake command during OpenVINO build.

> **Note**: Currently only Python 3.10 on Ubuntu 22.04 is verified. So the target device must have Python 3.10 in this case.

### RISC-V Emulation software
In order to test applications without hardware one can use emulation software. The command line example to launch executable file with riscv64 emulation:
```sh
<xuantie_install_path>/bin/qemu-riscv64 -cpu=<target_cpu> <executable_file_path>
```

For example, to emulate RVV 0.7.1:
```sh
<xuantie_install_path>/bin/qemu-riscv64 -cpu rv64,x-v=true,vext_spec=v0.7.1 <executable_file_path>
```

Or to emulate RVV 1.0:
```sh
<xuantie_install_path>/bin/qemu-riscv64 -cpu rv64,x-v=true,vext_spec=v1.0 <executable_file_path>
```

> **Note**: If you are using official `qemu` instead of modified version by Xuantie, you should specify the CPU model with `-cpu rv64,v=true,vext_spec=v1.0` (for `qemu` version greater than `8.0`). 

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

