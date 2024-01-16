# Building the OpenVINO™ Python API

**Refer to ["How to build OpenVINO" in OpenVINO™ developer documentation](../../../../docs/dev/build.md) for general building instructions.**

For each platform, you can build and install the API as a part of OpenVINO™ Toolkit or as a Python wheel.
A Python wheel is a portable package that allows you to install OpenVINO™ in either your Python distribution or a dedicated virtual environment.

## Virtual environments

OpenVINO can be built based on specific virtual environments such as [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) or [pyenv](https://github.com/pyenv/pyenv). It is highly recommended to use virtual environments during development. They improve development process and allow better management of Python versions and packages.

*Note: Supported Python versions can be found in ["System Requirements" section](../../../../docs/install_guides/pypi-openvino-dev.md#system-requirements).*

### Example: using pyenv with OpenVINO™ on Linux based system

1. First, set up the `pyenv` project. Please follow [official instructions of the pyenv project](https://github.com/pyenv/pyenv#installation) for any additional information.


2. Install a desired Python version. Following example will use Python in version 3.10.7. To correctly link libraries, an installed Python version must match OpenVINO™: 
    * Python with a shared library for a dynamically linked OpenVINO™:
    ```shell
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.7
    ```
    * Python with a static library version for a static build of OpenVINO™:
    ```shell
    pyenv install --verbose 3.10.7
    ```

3. Create a virtual environment based on the installed Python version:
    ```
    pyenv virtualenv 3.10.7 ov-py310
    ```

4. Activate the environment:
    ```bash
    pyenv activate ov-py310
    ```

5. Install developer requirements for OpenVINO™ Python API while inside virtual environment:
    ```shell
    cd <openvino_repo>
    pip install -r src/bindings/python/requirements.txt
    pip install -r src/bindings/python/requirements_test.txt
    ```
    If `-DENABLE_WHEEL=ON` flag is present in `cmake` command, additionally install wheel requirements:
    ```
    pip install -r src/bindings/python/wheel/requirements-dev.txt
    ```

6. Add following flags to the main `cmake` command to use specific virtual environment (requires cmake 3.16 and higher):
    ```shell
    -DPython3_EXECUTABLE=/home/user/.pyenv/versions/3.10.7/bin/python
    ```

7. Follow the rest of building and installation steps from ["How to build OpenVINO" developer documentation](../../../../docs/dev/build.md).

## Project dependencies management
For details please refer to [Python requirements and version constraints management](./requirements_management.md).

## Run tests to verify OpenVINO™ Python API

Follow instructions in [How to test OpenVINO™ Python API?](./test_examples.md#Running_OpenVINO™_Python_API_tests) to verify the build.

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
