# Build OpenVINO™ Runtime for Android systems

This article describes how to build Inference Engine for Android operating systems.

## Software requirements

- [CMake](https://cmake.org/download/) 3.13 or higher
- Android NDK (this guide has been validated with r20 release)

## How to build

1. Download and unpack [Android NDK](https://developer.android.com/ndk/downloads). Let's assume that `~/Downloads` is used as a working folder.
  ```sh
  cd ~/Downloads
  wget https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip

  unzip android-ndk-r20-linux-x86_64.zip
  mv android-ndk-r20 android-ndk
  ```

2. Create a build folder:
  ```sh
  mkdir build
  ```

3. Change working directory to `build` and run `cmake` to create makefiles. Then run `make`.
  ```sh
  cd build

  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=~/Downloads/android-ndk/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM=21 \
    -DANDROID_STL=c++_shared

  make --jobs=$(nproc --all)
  ```

  * `ANDROID_ABI` specifies target architecture:
    * `x86_64` for x64 build
    * `armeabi-v7a with NEON` for ARM with NEON support
    * `arm64-v8a` for ARM 64 bits
  * `ANDROID_PLATFORM` - Android API version
  * `ANDROID_STL` specifies that shared C++ runtime is used. Copy `~/Downloads/android-ndk/sources/cxx-stl/llvm-libc++/libs/x86_64/libc++_shared.so` from Android NDK along with built binaries

4. To reduce the binaries size, use `strip` tool from NDK:

```bash
~/Downloads/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/x86_64-linux-android/bin/strip openvino/bin/intel64/Release/lib/*.so
```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

