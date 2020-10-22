# Building the Python API for nGraph

You can build the nGraph Python API from sources by following instructions in this document. A Python wheel is a 
portable package which will allow you to install nGraph in your Python distribution, or dedicated virtual environment.

## Build nGraph Python Wheels on Linux or MacOS

### Prerequisites

In order to build the nGraph Python wheel, you will need to install a few packages.

On Ubuntu 20.04 LTS you can use the following instructions to install the required packages, including Python and Cython.

    apt install git wget build-essential cmake
    apt install python3 python3-dev python3-pip python3-virtualenv python-is-python3

You can see a full working example on an Ubuntu environment used in our continuous environment in this 
[Dockerfile](https://github.com/openvinotoolkit/openvino/blob/master/.ci/openvino-onnx/Dockerfile).

On MacOS you can use [Homebrew](https://brew.sh) to install required packages:

    brew install cmake
    brew install automake
    brew install libtool
    brew install python3

Install Cython in the Python installation, or virtualenv which you are planning to use:

    pip3 install cython

### Configure, build and install OpenVINO

The following section will illustrate how to download, build and install OpenVINO in a workspace directory specified
by the `${MY_OPENVINO_BASEDIR}` variable. Let's start by setting this variable to a directory of your choice: 

    export MY_OPENVINO_BASEDIR=/path/to/my/workspace

Now we can clone OpenVINO, configure it using `cmake` and build using `make`. Please note that we're disabling
the building of a few modules by setting the `ENABLE_*` flag to `OFF`. In order to build the OpenVINO Python APIs
set the mentioned flags to `ON`. Note the `CMAKE_INSTALL_PREFIX`, which defaults to `/usr/local/` if not set.

    cd "${MY_OPENVINO_BASEDIR}"
    git clone --recursive https://github.com/openvinotoolkit/openvino.git
    mkdir openvino/build
    cd openvino/build
    
    cmake .. \
        -DENABLE_CLDNN=OFF \
        -DENABLE_OPENCV=OFF \
        -DENABLE_VPU=OFF \
        -DENABLE_PYTHON=ON \
        -DNGRAPH_PYTHON_BUILD_ENABLE=ON \
        -DNGRAPH_ONNX_IMPORT_ENABLE=ON \
        -DCMAKE_INSTALL_PREFIX="${MY_OPENVINO_BASEDIR}/openvino_dist"
    
    make -j 4
    make install

If you would like to use a specific version of Python, or use a virtual environment you can set the `PYTHON_EXECUTABLE` 
variable. Examples: 

```
-DPYTHON_EXECUTABLE=/path/to/venv/bin/python
-DPYTHON_EXECUTABLE=$(which python3.8)
```

### Build nGraph Python wheel

When OpenVINO is built and installed, we can build the Python wheel by issuing the following command:

    make python_wheel

Once completed, the wheel package should be located under the following path:

    $ ls "${MY_OPENVINO_BASEDIR}/openvino/ngraph/python/dist/"
    ngraph_core-0.0.0-cp38-cp38-linux_x86_64.whl

You can now install the wheel in your Python environment:

    cd "${MY_OPENVINO_BASEDIR}/openvino/ngraph/python/dist/"
    pip3 install ngraph_core-0.0.0-cp38-cp38-linux_x86_64.whl

#### What does `make python_wheel` do?

The `python_wheel` target automates a few steps, required to build the wheel package. You can recreate the process 
manually by issuing the following commands: 

    cd "${MY_OPENVINO_BASEDIR}/openvino/ngraph/python"
    git clone --branch v2.5.0 https://github.com/pybind/pybind11.git
    export NGRAPH_CPP_BUILD_PATH="${MY_OPENVINO_BASEDIR}/openvino_dist"
    python3 setup.py bdist_wheel


## Build nGraph Python Wheels on Windows

### Prerequisites

In order to build OpenVINO and the nGraph Python wheel on Windows, you will need to install Visual Studio and Python. 

Once Python is installed, you will also need to install Cython using `pip install cython`.  

### Configure, build and install OpenVINO

Configure the build with a `cmake` invocation similar to the following. Note that you'll need to set the `-G` and 
`-DCMAKE_CXX_COMPILER` to match the version and location of your Visual Studio installation.

```
cmake .. ^
    -G"Visual Studio 16 2019" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="C:\temporary_install_dir" ^
    -DENABLE_CLDNN=OFF ^
    -DENABLE_OPENCV=OFF ^
    -DENABLE_VPU=OFF ^
    -DNGRAPH_PYTHON_BUILD_ENABLE=ON ^
    -DNGRAPH_ONNX_IMPORT_ENABLE=ON ^
    -DENABLE_PYTHON=ON ^
    -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"

```

There are a couple of things to notice here. One is that the full path to the x64 version of
MSVC compiler has to be specified. This is because DNNL requires a 64-bit version and cmake may
fail to detect it correctly.

The other equally important thing to note is that the temporary directory where the build is to be installed can be specified.
If the installation directory is not specified, the default location is `C:\Program Files\OpenVINO`.
This examples uses `C:\temporary_install_dir` however, a subdirectory of `openvino\build` works as well.
The final Python wheel will contain the contents of this temporary directory so it's important to set it.

If you want to specify an exact Python version, use the following options:
```
-DPYTHON_EXECUTABLE="C:\Program Files\Python37\python.exe" ^
-DPYTHON_LIBRARY="C:\Program Files\Python37\libs\python37.lib" ^
-DPYTHON_INCLUDE_DIR="C:\Program Files\Python37\include" ^
```

In order to build and install OpenVINO, build the `install` target:

    cmake --build . --target install --config Release -j 8

In this step OpenVINO will be built and installed to the directory specified above. You can
adjust the number of threads used in the building process to your machine's capabilities.

Build the Python wheel package:

    cmake --build . --target python_wheel --config Release -j 8

The final wheel should be located in `ngraph\python\dist` directory.

    dir openvino\ngraph\python\dist\
    10/09/2020  04:06 PM         4,010,943 ngraph_core-0.0.0-cp38-cp38-win_amd64.whl


## Run tests

### Using a virtualenv (optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Install the nGraph wheel and other requirements

    (venv) $ cd "${MY_OPENVINO_BASEDIR}/openvino/ngraph/python"
    (venv) $ pip3 install -r requirements.txt
    (venv) $ pip3 install -r requirements_test.txt
    (venv) $ pip3 install dist/ngraph_core-0.0.0-cp38-cp38-linux_x86_64.whl

### Run tests

You should now be able to run tests. 

You may need to run the `setupvars` script from OpenVINO to set paths to OpenVINO components.

    source ${MY_OPENVINO_BASEDIR}/openvino/scripts/setupvars/setupvars.sh

The minimum requirement is to set the `PYTHONPATH` to include the Inference Engine Python API: 

    export PYTHONPATH="${MY_OPENVINO_BASEDIR}/openvino/bin/intel64/Release/lib/python_api/python3.8/":${PYTHONPATH}
    pytest tests/

Now you can run tests using `pytest`:

    pytest tests
