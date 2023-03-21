# Build OpenVINO™ Runtime for Raspbian Stretch OS

> **NOTE**: [ARM CPU plugin](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/arm_plugin) are supported. The detailed instruction how to build ARM plugin is available in [OpenVINO contrib wiki](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin).

## Hardware Requirements
* Raspberry Pi 2 or 3 with Raspbian Stretch OS (32 or 64-bit).

  > **NOTE**: Despite the Raspberry Pi CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

You can compile the OpenVINO Runtime for Raspberry Pi in one of the two ways:
* [Native Compilation](#native-compilation), which is the simplest way, but time-consuming
* [Cross Compilation Using Docker](#cross-compilation-using-docker), which is the recommended way

## Native Compilation
Native compilation of the OpenVINO Runtime is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi 3.

1. Install dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake scons build-essential
  ```
2. Clone the repositories:
```
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino.git 
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino_contrib.git 
```
3. Go to the cloned `openvino` repository:

  ```bash
  cd openvino/
  ```
4. Create a build folder:

  ```bash
  mkdir build && cd build/
  ```
5. Build the OpenVINO Runtime:
* for MYRIAD support only:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DOPENVINO_EXTRA_MODULES=<OPENVINO_CONTRIB_PATH>/openvino_contrib/modules/arm_plugin \
        -DARM_COMPUTE_SCONS_JOBS=$(nproc --all) \
  .. && cmake --build . --parallel 
  ```

## Cross Compilation Using Docker

To cross-compile ARM CPU plugins using pre-configured `Dockerfile` you can use the following instruction: [Build OpenCV, OpenVINO™ and the plugin using pre-configured Dockerfile](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin#approach-1-build-opencv-openvino-and-the-plugin-using-pre-configured-dockerfile-cross-compiling-the-preferred-way).

## Additional Build Options

- To build Python API, install `libpython3-dev:armhf` and `python3-pip`
  packages using `apt-get`; then install `numpy` and `cython` python modules
  via `pip3`, adding the following options:
   ```sh
   -DENABLE_PYTHON=ON \
   -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
   -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
   ```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

