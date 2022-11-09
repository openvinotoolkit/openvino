## Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or later
- Microsoft\* Visual Studio 2015 or later on Windows\*
- gcc 4.8 or later on Linux
- Python 3.7 or higher on Linux\*
- Python 3.7 or higher on Windows\*

## Prerequisites

2. Install Inference Engine Python API dependencies:
```bash
pip3 install -r requirements.txt
```

## Building on Linux

Build Inference Engine Python API alongside with the Inference Engine build.
You need to run Inference Engine build with the following flags:

```shellscript
  cd <INSTALL_DIR>/openvino
  mkdir -p build
  cd build
  cmake -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=`which python3.7` \
  	-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
  	-DPYTHON_INCLUDE_DIR=/usr/include/python3.7 ..
  make -j16
```

## Building on Windows

You need to run Inference Engine build with the following flags:

```shellscript
	cd <INSTALL_DIR>/openvino
	mkdir build
	cd build
	set PATH=C:\Program Files\Python36\Scripts;%PATH%
	cmake -G "Visual Studio 15 2017 Win64" ^
		-DENABLE_PYTHON=ON ^
		-DPYTHON_EXECUTABLE="C:\Program Files\Python36\python.exe" ^
		-DPYTHON_INCLUDE_DIR="C:\Program Files\Python36\include" ^
		-DPYTHON_LIBRARY="C:\Program Files\Python36\libs\python36.lib" ..
```

Then build generated solution INFERENCE_ENGINE_DRIVER.sln using Microsoft\* Visual Studio or run `cmake --build . --config Release` to build from the command line.


## Building Python wheel
1) Install Inference Engine Python API dependencies:
```bash
pip3 install -r wheel/requirements-dev.txt
```
2) Install the *patchelf* tool on Linux only:
```shellscript
sudo apt install patchelf
```
3) Run Inference Engine build with the following options:

```shellscript
-DENABLE_PYTHON=ON
-DENABLE_WHEEL=ON
```

## Running sample

Before running the Python samples:
- add the folder with built `openvino` Python module (located at `bin/intel64/Release/lib/python_api/python3.7` for Linux) to the PYTHONPATH environment variable.
- add the folder with Inference Engine libraries to LD_LIBRARY_PATH variable on Linux (or PATH on Windows).

Example of command line to run classification sample:

```bash
python3 sample/classification_sample.py -m <path/to/xml> -i <path/to/input/image> -d CPU
```
