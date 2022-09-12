# How to build OpenVINO

The guide provides the basic information about the process of building OpenVINO.

```mermaid
gantt 
    %% Use a hack for centry as a persantage
    dateFormat YYYY
    axisFormat %y
    todayMarker off
    title       OpenVINO getting started pipeline
    Setup environment :env, 2000, 1716w
    Build openvino :crit, build, after env, 1716w
    Run tests :active, run, after build, 1716w
```

## Setup environment

This charpter describe how to prepare your machine for OpenVINO development.

### Software requirements 

For the OpenVINO build next tools are required:
<details><summary>Windows</summary>
<p>
    
- [CMake]\* 3.14 or higher
- Microsoft\* Visual Studio 2019, version 16.8 or later
- (Optional) Intel® Graphics Driver for Windows* (30.0) [driver package](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html).
- Python 3.6 or higher for OpenVINO Runtime Python API
- [Git for Windows*](https://gitforwindows.org/)
    
</p>
</details>
<details><summary>Linux</summary>
<p>

- [CMake] 3.13 or higher
- GCC 7.5 or higher to build OpenVINO Runtime
- Python 3.6 or higher for OpenVINO Runtime Python API
- (Optional) [Install Intel® Graphics Compute Runtime for OpenCL™ Driver package 19.41.14441](https://github.com/intel/compute-runtime/releases/tag/19.41.14441) to enable inference on Intel integrated GPUs.
    
<p>
</details>
<details><summary>Mac</summary>
<p>

- [CMake]\* 3.13 or higher
- Clang\* compiler from Xcode\* 10.1 or higher
- Python\* 3.6 or higher for the OpenVINO Runtime Python API
- libusb library (e.g., **brew install libusb**)
    
</p>
</details>

### Clone OpenVINO Repository

Clone OpenVINO repository and init submodules:
```sh
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
```

Congratulate! Now you are ready to build the OpenVINO.

## How to build

## Run application

[CMake]:https://cmake.org/download/
