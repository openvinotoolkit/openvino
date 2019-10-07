# Build Inference Engine

## Contents

- [Introduction](#introduction)
- [Build on Linux* Systems](#build-on-linux-systems)
  - [Software Requirements](#software-requirements)
  - [Build Steps](#build-steps)
  - [Additional Build Options](#additional-build-options)
- [(Optional) Use Custom OpenCV Builds for Inference Engine](#use-custom-opencv-builds-for-inference-engine)
- [Next Steps](#next-steps)
- [Additional Resources](#additional-resources)

## Introduction
The Inference Engine can infer models in different formats with various input and output formats.

The open source version of Inference Engine includes the following plugins:

| PLUGIN               | DEVICE TYPES |
| ---------------------| -------------|
| CPU plugin           | Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
| GPU plugin           | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
| GNA plugin           | Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel® Pentium® Silver processor J5005, Intel® Celeron® processor J4005, Intel® Core™ i3-8121U processor |
| MYRIAD plugin        | Intel® Movidius™ Neural Compute Stick powered by the Intel® Movidius™ Myriad™ 2, Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X |
| Heterogeneous plugin | Heterogeneous plugin enables computing for inference on one network on several Intel® devices. |

Please see additional document on the low-precision (int8) flow in the root directory.

## Build on Linux* Systems

The software was validated on:
- Ubuntu\* 16.04, 18.04 (64-bit) with default GCC

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.5 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- Python 2.7 or higher for Inference Engine Python API wrapper
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.04.12237](https://github.com/intel/compute-runtime/releases/tag/19.04.12237).

### Build Steps
1. Clone submodules:
    ```sh
    cd dldt/inference-engine
    git submodule init
    git submodule update --recursive
    ```
2. Install build dependencies using the `install_dependencies.sh` script in the project root folder.
3. By default, the build enables the Inference Engine GPU plugin to infer models on your Intel® Processor Graphics. This requires you to [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.04.12237](https://github.com/intel/compute-runtime/releases/tag/19.04.12237) before running the build. If you don't want to use the GPU plugin, use the `-DENABLE_CLDNN=OFF` CMake build option and skip the installation of the Intel® Graphics Compute Runtime for OpenCL™ Driver.
4. Create a build folder:
```sh
  mkdir build && cd build
```
5. Inference Engine uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make --jobs=$(nproc --all)
```

### Additional Build Options

You can use the following additional build options:

- Required versions of TBB/OMP and OpenCV packages are downloaded automatically by the CMake-based script. If you want to use the automatically downloaded packages but you already have installed TBB or OpenCV packages configured in your environment, you may need to clean the `TBBROOT` and `OpenCV_DIR` environment variables before running the `cmake` command, otherwise they won't be downloaded and the build may fail if incompatible versions were installed. 

- If the CMake-based build script can not find and download the OpenCV package that is supported on your platform, or if you want to use a custom build of the OpenCV library, refer to the [Use Custom OpenCV Builds](#use-custom-opencv-builds-for-inference-engine) section for details. 

- To build the Python API wrapper, use the `-DENABLE_PYTHON=ON` option. To specify an exact Python version, use the following options:
   ```sh
   -DPYTHON_EXECUTABLE=`which python3.7` \
   -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
   ```
- To switch off/on the CPU and GPU plugins, use the `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF` respectively.
  
5. Adding to your project

    For CMake projects, set an environment variable `InferenceEngine_DIR`:

    ```sh
    export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/
    ```

    Then you can find Inference Engine by `find_package`:

    ```cmake
    find_package(InferenceEngine)

    include_directories(${InferenceEngine_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)
    ```

## Next Steps

Congratulations, you have built the Inference Engine. To get started with the OpenVINO™ DLDT, proceed to the Get Started guides:

* [Get Started with Deep Learning Deployment Toolkit on Linux*](../get-started-linux.md)

## Additional Resources

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

---
\* Other names and brands may be claimed as the property of others.