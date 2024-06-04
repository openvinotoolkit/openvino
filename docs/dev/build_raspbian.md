# Build OpenVINO™ Runtime for Raspbian Stretch OS

> **NOTE**: Since 2023.0 release, you can compile [OpenVINO Intel CPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu) on ARM platforms.

## Hardware Requirements
* Raspberry Pi with Raspbian Stretch OS or Raspberry Pi OS (32 or 64-bit).

  > **NOTE**: Despite the Raspberry Pi CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

## Compilation
You can perform native compilation of the OpenVINO Runtime for Raspberry Pi, which is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi 3.

1. Install dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake scons build-essential
  ```
2. Clone the repository:
```
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino.git 
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
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DARM_COMPUTE_SCONS_JOBS=$(nproc --all) \
  .. && cmake --build . --parallel 
  ```

> **NOTE**: The build command may fail due to insufficient RAM. To fix this issue, you can increase the swap size:
1. Deactivate the current swap:
```bash
sudo dphys-swapfile swapoff
```
2. Modify the swap size by setting `CONF_SWAPSIZE=8192` in `/etc/dphys-swapfile`.
3. Recreate the swap file:
```bash
sudo dphys-swapfile setup
```
3. Start swap:
```bash
sudo dphys-swapfile swapon
```

## Additional Build Options

- To build Python API, install `libpython3-dev:armhf` and `python3-pip`
  packages using `apt-get`; add the following options:
   ```sh
   -DENABLE_PYTHON=ON \
   -DPython3_EXECUTABLE=/usr/bin/python3.8
   ```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

