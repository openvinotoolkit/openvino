# Build OpenVINO™ Runtime for Android systems

This article describes how to build OpenVINO for Android operating systems.

## Software requirements

- [CMake](https://cmake.org/download/) 3.13 or higher
- [SCons](https://scons.org/pages/download.html) 4.6.0 or higher
- [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools) (this guide has been validated with 30.0.0 release)
- [Android NDK](https://developer.android.com/ndk/downloads) (this guide has been validated with r26 release)

## How to build

### Create a work directory 
  ```sh
  mkdir openvino-android
  export OPV_HOME_DIR=${PWD}/openvino-android
  ```

### Download and unpack Android packages 
* Download and unpack [Android NDK](https://developer.android.com/ndk/downloads)
  ```sh
  wget https://dl.google.com/android/repository/android-ndk-r26d-linux.zip --directory-prefix $OPV_HOME_DIR

  unzip $OPV_HOME_DIR/android-ndk-r26d-linux.zip -d $OPV_HOME_DIR
  mv $OPV_HOME_DIR/android-ndk-r26d $OPV_HOME_DIR/android-ndk
  export ANDROID_NDK_PATH=$OPV_HOME_DIR/android-ndk
  ```
* Download and unpack [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools)
  ```sh
  wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip --directory-prefix $OPV_HOME_DIR

  unzip $OPV_HOME_DIR/platform-tools-latest-linux.zip -d $OPV_HOME_DIR
  export ANDROID_TOOLS_PATH=$OPV_HOME_DIR/platform-tools
  ```
_For Windows and Mac operating systems, the downloading and unpacking steps are similar._

### Set the environment variables for building
  ```sh
  # If you have no android devices please set CURRENT_ANDROID_ABI according to your preferences e.g. export CURRENT_ANDROID_ABI=arm64-v8a
  export CURRENT_ANDROID_ABI=`$ANDROID_TOOLS_PATH/adb shell getprop ro.product.cpu.abi`
  export CURRENT_ANDROID_PLATFORM=30
  export CURRENT_ANDROID_STL=c++_shared
  export CURRENT_CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake
  ```
* `ANDROID_ABI` specifies the target architecture:
    * `x86_64` for x64 build
    * `armeabi-v7a with NEON` for ARM with NEON support
    * `arm64-v8a` for ARM 64 bits
* `ANDROID_PLATFORM` specifies the Android API version.
* `ANDROID_STL` indicates that a shared C++ runtime is used.

### Build and install OneTBB™
To improve the parallelism performance of the OpenVINO™ library using OneTBB, it is required to separately build OneTBB for a specific version of the Android NDK:
  ```sh
  # Clone OneTBB™ repository 
  git clone --recursive https://github.com/oneapi-src/oneTBB $OPV_HOME_DIR/one-tbb
  # Create build and install directory 
  mkdir $OPV_HOME_DIR/one-tbb-build $OPV_HOME_DIR/one-tbb-install
  # Configure OneTBB™ CMake project 
  cmake -S $OPV_HOME_DIR/one-tbb \
        -B $OPV_HOME_DIR/one-tbb-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$OPV_HOME_DIR/one-tbb-install \
        -DCMAKE_TOOLCHAIN_FILE=$CURRENT_CMAKE_TOOLCHAIN_FILE \
        -DANDROID_ABI=$CURRENT_ANDROID_ABI \
        -DANDROID_PLATFORM=$CURRENT_ANDROID_PLATFORM \
        -DANDROID_STL=$CURRENT_ANDROID_STL \
        -DTBB_TEST=OFF \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--undefined-version" 
  # Build OneTBB™ project 
  cmake --build $OPV_HOME_DIR/one-tbb-build --parallel
  # Install OneTBB™ project 
  cmake --install $OPV_HOME_DIR/one-tbb-build
  ```

### Build and install OpenVINO™
  ```sh
  # Clone OpenVINO™ repository 
  git clone --recursive https://github.com/openvinotoolkit/openvino $OPV_HOME_DIR/openvino
  # Create build and install directory 
  mkdir $OPV_HOME_DIR/openvino-build $OPV_HOME_DIR/openvino-install
  # Configure OpenVINO™ CMake project 
  cmake -S $OPV_HOME_DIR/openvino \
        -B $OPV_HOME_DIR/openvino-build \
        -DCMAKE_INSTALL_PREFIX=$OPV_HOME_DIR/openvino-install \
        -DCMAKE_TOOLCHAIN_FILE=$CURRENT_CMAKE_TOOLCHAIN_FILE \
        -DANDROID_ABI=$CURRENT_ANDROID_ABI \
        -DANDROID_PLATFORM=$CURRENT_ANDROID_PLATFORM \
        -DANDROID_STL=$CURRENT_ANDROID_STL \
        -DTBB_DIR=$OPV_HOME_DIR/one-tbb-install/lib/cmake/TBB
  # Build OpenVINO™ project 
  cmake --build $OPV_HOME_DIR/openvino-build --parallel
  # Install OpenVINO™ project 
  cmake --install $OPV_HOME_DIR/openvino-build
  ```

### Download the example model
  We will use `mobelinet-v3-tf` example model
  ```sh
  mkdir $OPV_HOME_DIR/mobelinet-v3-tf
  wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.xml -P $OPV_HOME_DIR/mobelinet-v3-tf/
  wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.bin -P $OPV_HOME_DIR/mobelinet-v3-tf/
  ```

### Use the ADB tool to run the example model and check the library
_This example is demonstrated for aarch64 architecture_
  ```sh
  # Copy OpenVINO™ libraries to android device
  $ANDROID_TOOLS_PATH/adb push --sync $OPV_HOME_DIR/openvino-install/runtime/lib/aarch64/* /data/local/tmp/
  # Copy OneTBB libraries to android device
  $ANDROID_TOOLS_PATH/adb push --sync $OPV_HOME_DIR/one-tbb-install/lib/* /data/local/tmp/
  # Copy shared STL library to android device
  $ANDROID_TOOLS_PATH/adb push --sync $ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/
  # Copy example model files
  $ANDROID_TOOLS_PATH/adb push --sync $OPV_HOME_DIR/mobelinet-v3-tf /data/local/tmp/
  # Copy OpenVINO™ benchmark_app tool to android device
  $ANDROID_TOOLS_PATH/adb push --sync $OPV_HOME_DIR/openvino/bin/aarch64/Release/benchmark_app /data/local/tmp/
  # Run OpenVINO™ benchmark_app tool
  $ANDROID_TOOLS_PATH/adb shell "LD_LIBRARY_PATH=/data/local/tmp ./data/local/tmp/benchmark_app -m /data/local/tmp/mobelinet-v3-tf/v3-small_224_1.0_float.xml -hint latency"
  ```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)
