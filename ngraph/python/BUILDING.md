# Building the Python API for nGraph

## Building nGraph Python Wheels

If you want to try a newer version of nGraph's Python API than is available
from PyPI, you can build the latest version from source code. This process is
very similar to what is outlined in our [ngraph_build] instructions with two
important differences:

1. You must specify: `-DNGRAPH_PYTHON_BUILD_ENABLE=ON` and `-DNGRAPH_ONNX_IMPORT_ENABLE=ON`
   when running `cmake`.

2. Instead of running `make`, use the command `make python_wheel`.

    `$ cmake ../ -DNGRAPH_PYTHON_BUILD_ENABLE=ON -DNGRAPH_ONNX_IMPORT_ENABLE=ON`

    `$ make python_wheel`

After this procedure completes, the `ngraph/build/python/dist` directory should
contain the Python packages of the version you cloned. For example, if you
checked out and built `0.21` for Python 3.7, you might see something like:

    $ ls python/dist/
    ngraph-core-0.21.0rc0.tar.gz
    ngraph_core-0.21.0rc0-cp37-cp37m-linux_x86_64.whl

## Building nGraph Python Wheels on Windows

The build process on Windows consists of 3 steps:

1. Configure the build with the following `cmake` invocation:
~~~~
cmake ..
      -G"Visual Studio 15 2017 Win64"
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX="C:\temporary_install_dir"
      -DNGRAPH_PYTHON_BUILD_ENABLE=TRUE
      -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
      -DCMAKE_CXX_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64
~~~~
There are a couple of things to notice here. One is that the full path to the x64 version of
MSVC compiler has to be specified. This is because DNNL requires a 64-bit version and cmake may
fail to detect it correctly.
The other equally important thing to note is that the temporary directory where the build is to be installed can be specified.
This examples uses `C:\temporary_install_dir` however, a subdirectory of `ngraph\build` works as well.
The final Python wheel will contain the contents of this temporary directory so it's very important to set it.

2. Build the `install` target:

    `cmake --build . --target install --config Release -j 8`

In this step nGraph will be built and installed to the temporary directory specified above. You can
adjust the number of threads used in the building process to your machine's capabilities.

3. Build the Python wheel itself:

    `cmake --build . --target python_wheel --config Release -j 8`

The final wheel should be located in `build\python\dist` directory.

### Using a virtualenv (optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Installing the wheel

You may wish to use a virutualenv for your installation.

    (venv) $ pip install ngraph/build/python/dist/ngraph_core-0.21.0rc0-cp37-cp37m-linux_x86_64.whl

## Running tests

Unit tests require additional packages be installed:

    (venv) $ cd ngraph/python
    (venv) $ pip install -r test_requirements.txt

Then run tests:

    (venv) $ pytest test/ngraph/

[ngraph_build]: http://ngraph.nervanasys.com/docs/latest/buildlb.html
