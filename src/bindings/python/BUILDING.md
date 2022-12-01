# Building the OpenVINO™ Python API

This document provides links to the instructions for building the OpenVINO™ Python API from source.

For each platform, you can build and install the API as a part of OpenVINO™ Toolkit or as a Python wheel.
A Python wheel is a portable package that allows you to install OpenVINO™ in either your Python distribution or a dedicated virtual environment.

*Note: If changes are adding new compilation units, files or change CMake scripts, there is a need to remove the exisiting build and rebuild it from scratch!*

- ### Linux* Systems
    Follow instructions available on [GitHub wiki pages for Linux build](https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinux).

    Exact Python instructions can be found in [Additional Build Options section](https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinux#additional-build-options).

- ### Windows* Systems
    Follow instructions available on [GitHub wiki pages for Windows build](https://github.com/openvinotoolkit/openvino/wiki/BuildingForWindows).

    Exact Python instructions can be found in [Additional Build Options section](https://github.com/openvinotoolkit/openvino/wiki/BuildingForWindows#additional-build-options).

- ### macOS* Systems for Intel CPU
    Follow instructions available on [GitHub wiki pages for macOS build](https://github.com/openvinotoolkit/openvino/wiki/BuildingForMacOS_x86_64).

- ### macOS* Systems for Apple Silicon
    Follow instructions available on [GitHub wiki pages for macOS build](https://github.com/openvinotoolkit/openvino/wiki/BuildingForMacOS_arm64).

- ### Raspbian Stretch* OS
    Follow instructions available on [GitHub wiki pages for Raspbian Stretch OS build](https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS).

    Exact Python instructions can be found in [Additional Build Options section](https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS#additional-build-options).

## Run tests to verify OpenVINO™ Python API

Follow instructions in [How to test OpenVINO™ Python API?](./docs/test_examples.md#Running_OpenVINO™_Python_API_tests) to verify the build.

*Note: You may need to run the `setupvars` script from the OpenVINO™ Toolkit to set paths to OpenVINO™ components.*
