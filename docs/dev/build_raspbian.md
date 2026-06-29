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

## Cross-compilation for Raspberry Pi ARM64

Cross-compilation uses one host-specific CMake toolchain file:

* `cmake/arm64.toolchain.cmake` for Linux-hosted cross-compilation to Raspberry Pi AArch64 Linux;
* `cmake/rpi-aarch64-linux-from-macos.toolchain.cmake` for macOS-hosted cross-compilation to Raspberry Pi AArch64 Linux;
* `cmake/rpi-aarch64-linux-from-windows.toolchain.cmake` for Windows-hosted cross-compilation to Raspberry Pi AArch64 Linux.

The macOS and Windows toolchains generate compiler, binutils, and Arm Compute Library SCons wrappers in the build directory. Wrapper templates and helper sources are stored in `cmake/rpi_cross/`.

Create a working directory for downloaded host tools and build outputs:

```bash
mkdir openvino-rpi-cross
export OV_RPI_HOME=${PWD}/openvino-rpi-cross
```

On Windows PowerShell:

```powershell
$env:OV_RPI_HOME = "$PWD\openvino-rpi-cross"
New-Item -ItemType Directory -Force -Path $env:OV_RPI_HOME
```

### Prepare a target sysroot

The macOS and Windows toolchains require a Raspberry Pi ARM64 Linux sysroot. The sysroot must match the Raspberry Pi userspace you will run on, especially glibc and C++ runtime versions. Use a reproducible package repository, OS image, or the sysroot bundled with the installed cross toolchain; do not copy files from a live Raspberry Pi.

The examples below use the sysroot bundled with the selected cross toolchain. If you use a distro-matched sysroot from another source, pass it with `-DOV_RPI_SYSROOT=<path>`.

### Linux host

Install host tools:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build scons python3 python3-pip \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu pkg-config xz-utils zstd
```

Configure and build:

```bash
cmake -S . -B build-rpi-aarch64 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/arm64.toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=$PWD/install-rpi-aarch64

cmake --build build-rpi-aarch64 --parallel
cmake --install build-rpi-aarch64
```

The Linux toolchain file uses the standard `aarch64-linux-gnu-*` tool names and is intentionally kept minimal. If your Linux cross toolchain needs an explicit target sysroot, pass standard CMake sysroot settings that match your distribution toolchain.

### macOS host

Install host tools and download the Linux AArch64 GNU cross toolchain:

```bash
brew install cmake ninja scons python xz zstd
brew install messense/macos-cross-toolchains/aarch64-unknown-linux-gnu
```

Set the toolchain sysroot for CMake:

```bash
export OV_RPI_SYSROOT="$(aarch64-linux-gnu-gcc -print-sysroot)"
export OV_RPI_TOOLCHAIN_PREFIX=aarch64-linux-gnu
```

Configure and build with the macOS-hosted Raspberry Pi ARM64 Linux toolchain:

```bash
cmake -S . -B build-rpi-aarch64-macos -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/rpi-aarch64-linux-from-macos.toolchain.cmake \
    -DOV_RPI_TOOLCHAIN_PREFIX=$OV_RPI_TOOLCHAIN_PREFIX \
    -DOV_RPI_SYSROOT=$OV_RPI_SYSROOT \
    -DCMAKE_INSTALL_PREFIX=$PWD/install-rpi-aarch64-macos

cmake --build build-rpi-aarch64-macos --parallel
cmake --install build-rpi-aarch64-macos
```

The macOS toolchain file generates host wrapper scripts in the build directory and points CMake, binutils, and Arm Compute Library at the target sysroot.

### Windows host

Download and unpack CMake and Ninja from PowerShell:

```powershell
$env:OV_RPI_TOOLS = "$env:OV_RPI_HOME\tools"
New-Item -ItemType Directory -Force -Path $env:OV_RPI_TOOLS

$CMakeVersion = "3.30.5"
$CMakeUrl = "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/cmake-$CMakeVersion-windows-x86_64.zip"
$CMakeZip = "$env:OV_RPI_HOME\cmake-$CMakeVersion-windows-x86_64.zip"
Invoke-WebRequest -Uri $CMakeUrl -OutFile $CMakeZip
Expand-Archive -Path $CMakeZip -DestinationPath $env:OV_RPI_TOOLS -Force
Move-Item -Force "$env:OV_RPI_TOOLS\cmake-$CMakeVersion-windows-x86_64" "$env:OV_RPI_TOOLS\cmake"

$NinjaVersion = "1.12.1"
$NinjaUrl = "https://github.com/ninja-build/ninja/releases/download/v$NinjaVersion/ninja-win.zip"
$NinjaZip = "$env:OV_RPI_HOME\ninja-win.zip"
New-Item -ItemType Directory -Force -Path "$env:OV_RPI_TOOLS\ninja"
Invoke-WebRequest -Uri $NinjaUrl -OutFile $NinjaZip
Expand-Archive -Path $NinjaZip -DestinationPath "$env:OV_RPI_TOOLS\ninja" -Force
```

Install Python and SCons:

```powershell
$PythonVersion = "3.12"
winget install --id "Python.Python.$PythonVersion" -e
py "-$PythonVersion" -m pip install --user scons
$PythonScripts = py "-$PythonVersion" -c "import site; print(site.getuserbase() + r'\Scripts')"
```

Download and unpack the Arm GNU Toolchain for Windows targeting AArch64 GNU/Linux:

```powershell
$ToolchainUrl = "https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-mingw-w64-i686-aarch64-none-linux-gnu.zip"
$ToolchainZip = "$env:OV_RPI_HOME\arm-gnu-toolchain-aarch64-none-linux-gnu.zip"
Invoke-WebRequest -Uri $ToolchainUrl -OutFile $ToolchainZip
Expand-Archive -Path $ToolchainZip -DestinationPath $env:OV_RPI_TOOLS -Force
$env:OV_RPI_TOOLCHAIN_ROOT = "$env:OV_RPI_TOOLS\arm-gnu-toolchain-14.3.rel1-mingw-w64-i686-aarch64-none-linux-gnu"
```

The Windows-hosted Arm package uses the `aarch64-none-linux-gnu` prefix and provides the target sysroot under `$env:OV_RPI_TOOLCHAIN_ROOT\aarch64-none-linux-gnu\libc`.

Configure and build with the Windows-hosted Raspberry Pi ARM64 Linux toolchain:

```powershell
$env:OV_RPI_REPO = (Get-Location).Path
$env:Path = "$env:OV_RPI_TOOLCHAIN_ROOT\bin;$env:OV_RPI_TOOLS\ninja;$env:OV_RPI_TOOLS\cmake\bin;$PythonScripts;$env:Path"
$env:OV_RPI_SYSROOT = "$env:OV_RPI_TOOLCHAIN_ROOT\aarch64-none-linux-gnu\libc"
$env:OV_RPI_TOOLCHAIN_PREFIX = "aarch64-none-linux-gnu"

& "$env:OV_RPI_TOOLS\cmake\bin\cmake.exe" -S $env:OV_RPI_REPO -B "$env:OV_RPI_REPO\build-rpi-aarch64-windows" -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$env:OV_RPI_TOOLS\ninja\ninja.exe" `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$env:OV_RPI_REPO\cmake\rpi-aarch64-linux-from-windows.toolchain.cmake" `
    "-DOV_RPI_TOOLCHAIN_PREFIX=$env:OV_RPI_TOOLCHAIN_PREFIX" `
    "-DOV_RPI_SYSROOT=$env:OV_RPI_SYSROOT" `
    "-DCMAKE_INSTALL_PREFIX=$env:OV_RPI_REPO\install-rpi-aarch64-windows"

& "$env:OV_RPI_TOOLS\cmake\bin\cmake.exe" --build "$env:OV_RPI_REPO\build-rpi-aarch64-windows" --parallel
& "$env:OV_RPI_TOOLS\cmake\bin\cmake.exe" --install "$env:OV_RPI_REPO\build-rpi-aarch64-windows"
```

The Windows toolchain file generates host wrapper scripts, a local `grep.exe` required by OpenVINO's CMake probes, and Arm Compute Library SCons wrappers in the build directory. Do not mix the Arm GNU Toolchain C++ runtime headers with a different distro sysroot unless the libc and runtime versions are known to be compatible.

### Build CPU benchmark runtime targets

To build only the runtime pieces needed to run `benchmark_app` on the Raspberry Pi CPU, build the CPU plugin, the IR frontend, and `benchmark_app` explicitly:

```bash
cmake --build build-rpi-aarch64 --target \
    openvino_intel_cpu_plugin openvino_ir_frontend benchmark_app \
    --parallel
```

For macOS and Windows builds, replace `build-rpi-aarch64` with `build-rpi-aarch64-macos` or `build-rpi-aarch64-windows`.

### Verify and run on Raspberry Pi

After any cross-build, verify that the produced binaries are Linux AArch64 ELF files:

```bash
${OV_RPI_TOOLCHAIN_PREFIX:-aarch64-linux-gnu}-readelf -h \
    bin/aarch64/Release/benchmark_app
```

On a Windows host, use the Arm GNU Toolchain `readelf.exe`:

```powershell
& "$env:OV_RPI_TOOLCHAIN_ROOT\bin\aarch64-none-linux-gnu-readelf.exe" -h `
    .\bin\aarch64\Release\benchmark_app
```

The output must include `Class: ELF64` and `Machine: AArch64`.

For a quick CPU smoke test, copy the runtime output directory to the Raspberry Pi and run `benchmark_app` with the copied libraries on `LD_LIBRARY_PATH`:

```bash
rsync -a bin/aarch64/Release/ rpi@raspberrypi:~/openvino-rpi-cross/bin/

ssh rpi@raspberrypi \
    'cd ~/openvino-rpi-cross/bin && \
     LD_LIBRARY_PATH=$PWD ./benchmark_app \
       -m ~/models/model.xml \
       -d CPU \
       -api sync \
       -niter 10 \
       -nireq 1 \
       -nstreams 1 \
       -hint none'
```

Installing the build with `cmake --install` is the preferred way to preserve the expected runtime layout.

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

