## Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or later
- Microsoft\* Visual Studio 2015 or later on Windows\*
- gcc 4.8 or later on Linux
- Python 2.7 or higher on Linux\*
- Python 3.4 or higher on Windows\*

## Prerequisites

Install the following Python modules:
- opencv-python
- numpy
- cython

## Building on Windows
```shellscript
	mkdir build
	cd build
	set PATH=C:\Program Files\Python36\Scripts;%PATH%
	cmake -G "Visual Studio 14 2015 Win64" -DInferenceEngine_DIR=..\..\..\build ^
		-DPYTHON_EXECUTABLE="C:\Program Files\Python36\python.exe" ^
		-DPYTHON_INCLUDE_DIR="C:\Program Files\Python36\include" ^
		-DPYTHON_LIBRARY="C:\Program Files\Python36\libs\python36.lib" ..
```

Then build generated solution INFERENCE_ENGINE_DRIVER.sln using Microsoft\* Visual Studio.

## Building on Linux

```shellscript
  mkdir -p build
  cd build
  cmake -DInferenceEngine_DIR=../../../build -DPYTHON_EXECUTABLE=`which python3.6` \
  	-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
  	-DPYTHON_INCLUDE_DIR=/usr/include/python3.6 ..
  make -j16
```

Note: -DInferenceEngine_DIR parameter is needed to specify the folder with generated make files or Visual Studio solution used to build Inference Engine (see readme file in the inference-engine root folder).