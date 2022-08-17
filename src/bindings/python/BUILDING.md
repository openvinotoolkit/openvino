# Building the OpenVINO™ Python* API

This document provides the instructions for building the OpenVINO™ Python API from source on Linux, macOS and Windows 10 platforms.

For each platform, you can build and install the API as a part of OpenVINO™ Toolkit or as a Python wheel.
A Python wheel is a portable package that allows you to install OpenVINO™ in either your Python distribution or a dedicated virtual environment.

## Linux* and macOS*

### Prerequisites

To build the OpenVINO™ Python API, you need to install a few additional packages.

On Ubuntu* 20.04 LTS you can use the following instructions to install the required packages, including Python and Cython.

```bash
apt install git wget build-essential cmake
apt install python3 python3-dev python3-pip python3-virtualenv python-is-python3
```

On macOS, you can use [Homebrew](https://brew.sh) to install required packages:

```bash
brew install cmake
brew install python3
```

Install Cython in the Python installation, or virtualenv that you are planning to use:

```bash
pip3 install cython
```

### Configure and Build as a part of OpenVINO™ Toolkit on Linux and macOS

The following section illustrates how to build and install OpenVINO™ in a workspace directory using CMake.
The workspace directory is specified by the `${OV_WORKSPACE}` variable. Set this variable to a directory of your choice:

```bash
export OV_WORKSPACE=/path/to/my/workspace
```

Now you can clone the OpenVINO™ repository, configure it using `cmake` and build using `make`.

In order to build the OpenVINO™ Python APIs set the `ENABLE_PYTHON` flag to `ON`.

Note the `CMAKE_INSTALL_PREFIX`, which defaults to `/usr/local/` if not set.

```bash
cd ${OV_WORKSPACE}
git clone --recursive https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init

### You may also want to install OpenVINO™ dependencies with:
# ./install_build_dependencies.sh

mkdir build && cd build

cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_PYTHON=ON \
-DCMAKE_INSTALL_PREFIX="${OV_WORKSPACE}/openvino_dist"

make -j 4 install
```

The Python module is installed in the folder:

    ${OV_WORKSPACE}/openvino_dist/python/python<version>/

You may also find it in:

    ${OV_WORKSPACE}/openvino/bin/intel64/[BUILD_TYPE]/python_api/python<version>/

If you would like to use a specific version of Python, or use a virtual environment, you can add the `PYTHON_EXECUTABLE` variable to your CMake command line. For example:

```bash
-DPYTHON_EXECUTABLE=/path/to/venv/bin/python
# or while in virtual environment
-DPYTHON_EXECUTABLE=`which python`
```

Set up the OpenVINO™ environment in order to add the module path to `PYTHONPATH`:

```bash
source ${OV_WORKSPACE}/openvino_dist/setupvars.sh
```

If you want more control over enviroment variables, to simply enable OpenVINO™ Python API, export these variables:

```bash
export LD_LIBRARY_PATH=${OV_WORKSPACE}/openvino_dist/runtime/lib/intel64/
export PYTHONPATH=${OV_WORKSPACE}/openvino_dist/python/python<version>/
```

### Rebuild OpenVINO™ project

To rebuild project, simply navigate to build folder and re-run `make` command:
```bash
cd ${OV_WORKSPACE}/openvino/build

make -j 4 install
```
*Note: If changes are adding new compilation units, files or change CMake scripts, there is a need to remove the exisiting build and rebuild it from scratch!*

### Build an OpenVINO™ Python Wheel on Linux and macOS

#### First method - using CMake command
You can build the Python wheel by adding `ENABLE_WHEEL` flag to `cmake` command:

```bash
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_PYTHON=ON \
-DENABLE_WHEEL=ON \
-DCMAKE_INSTALL_PREFIX="${OV_WORKSPACE}/openvino_dist"
```

Once completed, the wheel package should be located under the following path:

```bash
$ ls ${OV_WORKSPACE}/openvino/src/bindings/python/dist
openvino-0.0.0-cp<version>-cp<version>-linux_x86_64.whl
```

You can now install the wheel in your Python environment:

```bash
cd ${OV_WORKSPACE}/openvino/src/bindings/python/dist
pip3 install openvino-0.0.0-cp<version>-cp<version>-linux_x86_64.whl
```

#### Second method - using setup.py file
You can build the Python wheel running the following command:

```bash
cd ${OV_WORKSPACE}/openvino/src/bindings/python/wheel
python3 setup.py bdist_wheel
```

Once completed, the wheel package should be located under the following path:

```bash
$ ls ${OV_WORKSPACE}/openvino/src/bindings/python/wheel/dist
openvino-0.0.0-cp<version>-cp<version>-linux_x86_64.whl
```

You can now install the wheel in your Python environment:

```bash
cd ${OV_WORKSPACE}/openvino/src/bindings/python/wheel/dist
pip3 install openvino-0.0.0-cp<version>-cp<version>-linux_x86_64.whl
```

## Windows* 10

### Prerequisites

In order to build OpenVINO™ and the nGraph Python wheel on Windows, you need to install Microsoft Visual Studio* and Python.

Once Python is installed, you also need to install Cython using `pip install cython`.

### Configure and Build as a Part of OpenVINO™ Toolkit on Windows

The following section illustrates how to build and install OpenVINO™ in a workspace directory using CMake.
The workspace directory is specified by the `OPENVINO_BASEDIR` variable. Set this variable to a directory of your choice:

    set OPENVINO_BASEDIR=/path/to/my/workspace

Configure the build with a `cmake` invocation similar to the following. Note that need to set `-G` and
`-DCMAKE_CXX_COMPILER` to match the version and location of your Microsoft Visual Studio installation.

```
cmake .. ^
    -G"Visual Studio 16 2019" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%OPENVINO_BASEDIR%/openvino_dist" ^
    -DENABLE_INTEL_GPU=OFF ^
    -DENABLE_INTEL_MYRIAD_COMMON=OFF ^
    -DENABLE_OV_ONNX_FRONTEND=ON ^
    -DENABLE_PYTHON=ON ^
    -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"

```

There are a couple of things to notice here. One is that the full path to the x64 version of
MSVC compiler has to be specified. This is because DNNL requires a 64-bit version and `cmake` may
fail to detect it correctly.

If you want to specify an exact Python version, use the following options:
```
-DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
-DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
-DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include" ^
```

In order to build and install OpenVINO™, build the `install` target:

    cmake --build . --target install --config Release -j 8

In this step, OpenVINO™ is built and installed to the directory specified above. You can
adjust the number of threads used in the building process to your machine's capabilities.

Set up the OpenVINO™ environment in order to add a module path to `PYTHONPATH`:

    %OPENVINO_BASEDIR%\openvino_dist\setupvars.bat

### Build an nGraph Python Wheel on Windows

Build the Python wheel package:

    cd "%OPENVINO_BASEDIR%/openvino/ngraph/python"
    python setup.py bdist_wheel

The final wheel should be located in the `ngraph\python\dist` directory.

    dir openvino\ngraph\python\dist\
    10/09/2020  04:06 PM         4,010,943 ngraph_core-0.0.0-cp38-cp38-win_amd64.whl

## Run Tests

### Use a virtualenv (Optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Install the nGraph Wheel and Other Requirements

    (venv) $ cd "${OPENVINO_BASEDIR}/openvino/ngraph/python"
    (venv) $ pip3 install -r requirements.txt
    (venv) $ pip3 install -r requirements_test.txt
    (venv) $ pip3 install dist/ngraph_core-0.0.0-cp38-cp38-linux_x86_64.whl

### Run Tests

You should now be able to run tests.

You may need to run the `setupvars` script from the OpenVINO™ Toolkit to set paths to OpenVINO™ components.

    source ${OPENVINO_BASEDIR}/openvino_dist/setupvars.sh

Now you can run tests using `pytest`:

    pytest tests
