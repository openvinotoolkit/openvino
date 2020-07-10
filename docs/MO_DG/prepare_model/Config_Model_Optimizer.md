# Configuring the Model Optimizer {#openvino_docs_MO_DG_prepare_model_Config_Model_Optimizer}

You must configure the Model Optimizer for the framework that was used to train
the model. This section tells you how to configure the Model Optimizer either
through scripts or by using a manual process.

## Using Configuration Scripts

You can either configure all three frameworks at the same time or install an
individual framework. The scripts delivered with the tool install all required
dependencies and provide the fastest and easiest way to configure the Model
Optimizer.

To configure all three frameworks, go to the
`<INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites`
directory and run:

*   For Linux\* OS:
```
install_prerequisites.sh
```
> **NOTE**: This command installs prerequisites globally. If you want to keep Model Optimizer in a separate sandbox, run the following commands instead:
```
virtualenv --system-site-packages -p python3 ./venv
```
```
source ./venv/bin/activate  # sh, bash, ksh, or zsh
```
```
./install_prerequisites.sh
```


*   For Windows\* OS:
```
install_prerequisites.bat
```

To configure a specific framework, go to the
`<INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites`
directory and run:

*   For Caffe\* on Linux:
```
install_prerequisites_caffe.sh
```
*   For Caffe on Windows:
```
install_prerequisites_caffe.bat
```
*   For TensorFlow\* on Linux:
```
install_prerequisites_tf.sh
```
*   For TensorFlow on Windows:
```
install_prerequisites_tf.bat
```
*   For MXNet\* on Linux:
```
install_prerequisites_mxnet.sh
```
*   For MXNet on Windows:
```
install_prerequisites_mxnet.bat
```
*   For Kaldi\* on Linux:
```
install_prerequisites_kaldi.sh
```
*   For Kaldi on Windows:
```
install_prerequisites_kaldi.bat
```
*   For ONNX\* on Linux:
```
install_prerequisites_onnx.sh
```
*   For ONNX on Windows:
```
install_prerequisites_onnx.bat
```

> **IMPORTANT**: **ONLY FOR CAFFE\*** By default, you do not need to install Caffe to create an
> Intermediate Representation for a Caffe model, unless you use Caffe for
> custom layer shape inference and do not write Model Optimizer extensions.
> To learn more about implementing Model Optimizer custom operations and the
> limitations of using Caffe for shape inference, see
> [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md).

## Using Manual Configuration Process

If you prefer, you can manually configure the Model Optimizer for one
framework at a time.

1.  Go to the Model Optimizer directory:
```shell
cd <INSTALL_DIR>/deployment_tools/model_optimizer/
```
2.  **Strongly recommended for all global Model Optimizer dependency installations**:
    Create and activate a virtual environment. While not required, this step is
    strongly recommended since the virtual environment creates a Python\*
    sandbox, and dependencies for the Model Optimizer do not influence the
    global Python configuration, installed libraries, or other components.
    In addition, a flag ensures that system-wide Python libraries are available
    in this sandbox. Skip this step only if you do want to install all the Model
    Optimizer dependencies globally:
    *   Create a virtual environment:
```shell
virtualenv -p /usr/bin/python3.6 .env3 --system-site-packages
```
    *   Activate the virtual environment:
```shell
virtualenv -p /usr/bin/python3.6 .env3/bin/activate
```
3.  Install all dependencies or only the dependencies for a specific framework:
    *   To install dependencies for all frameworks:
```shell
pip3 install -r requirements.txt
```
    *   To install dependencies only for Caffe:
```shell
pip3 install -r requirements_caffe.txt
```
    *   To install dependencies only for TensorFlow:
```shell
pip3 install -r requirements_tf.txt
```
    *   To install dependencies only for MXNet:
```shell
pip3 install -r requirements_mxnet.txt
```
    *   To install dependencies only for Kaldi:
```shell
pip3 install -r requirements_kaldi.txt
```
    *   To install dependencies only for ONNX:
```shell
pip3 install -r requirements_onnx.txt
```

## Using the protobuf Library in the Model Optimizer for Caffe\*

These procedures require:

*   Access to GitHub and the ability to use git commands
*   Microsoft Visual Studio\* 2013 for Win64\*
*   C/C++

Model Optimizer uses the protobuf library to load trained Caffe models.
By default, the library executes pure Python\* language implementation,
which is slow. These steps show how to use the faster C++ implementation
of the protobuf library on Windows OS or Linux OS.

### Using the protobuf Library on Linux\* OS

To use the C++ implementation of the protobuf library on Linux, it is enough to
set up the environment variable:
```sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
```

### <a name="protobuf-install-windows"></a>Using the protobuf Library on Windows\* OS

On Windows, pre-built protobuf packages for Python versions 3.4, 3.5, 3.6,
and 3.7 are provided with the installation package and can be found in
the
`<INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites`
folder. Please note that they are not installed with the
`install_prerequisites.bat` installation script due to possible issues
with `pip`, and you can install them at your own discretion. Make sure
that you install the protobuf version that matches the Python version
you use:

-   `protobuf-3.6.1-py3.4-win-amd64.egg` for Python 3.4
-   `protobuf-3.6.1-py3.5-win-amd64.egg` for Python 3.5
-   `protobuf-3.6.1-py3.6-win-amd64.egg` for Python 3.6
-   `protobuf-3.6.1-py3.7-win-amd64.egg` for Python 3.7

To install the protobuf package:

1. Open the command prompt as administrator.
2. Go to the `install_prerequisites` folder of the OpenVINO toolkit installation directory:
```sh
cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
```

3. Run the following command to install the protobuf for Python 3.6. If
   you want to install the protobuf for Python 3.4, 3.5, or 3.7, replace
   `protobuf-3.6.1-py3.6-win-amd64.egg` with the corresponding file
   name from the list above.
```sh
python -m easy_install protobuf-3.6.1-py3.6-win-amd64.egg
```
   If the Python version you use is lower than 3.4, you need to update
   it or <a href="#build-protobuf">build the library manually</a>.

#### <a name="build-protobuf"></a>Building the protobuf Library on Windows\* OS

> **NOTE**: These steps are optional. If you use Python version 3.4, 3.5, 3.6, or 3.7,
> you can <a href="#protobuf-install-windows">install the protobuf library</a> using the pre-built packages.

To compile the protobuf library from sources on Windows OS, do the following:

1.  Clone protobuf source files from GitHub:
```shell
git clone https://github.com/google/protobuf.git
cd protobuf
```
2.  Create a Visual Studio solution file. Run these commands:
```shell
cd C:\Path\to\protobuf\cmake\build
mkdir solution
cd solution C:\Path\to\protobuf\cmake\build\solution
cmake -G "Visual Studio 12 2013 Win64" ../..
```
3.  Change the runtime library option for `libprotobuf` and `libprotobuf-lite`:

   *   Open the project's **Property Pages** dialog box
   *   Expand the **C/C++** tab
   *   Select the **Code Generation** property page
   *   Change the **Runtime Library** property to **Multi-thread DLL (/MD)**
4.  Build the `libprotoc`, `protoc`, `libprotobuf`, and `libprotobuf-lite` projects in the **Release** configuration.
5.  Add a path to the build directory to the `PATH` environment variable:
```shell
set PATH=%PATH%;C:\Path\to\protobuf\cmake\build\solution\Release
```
6.  Go to the `python` directory:
```shell
cd C:\Path\to\protobuf\python
```
7.  Use a text editor to open and change these `setup.py` options:

   *   Change from <code>​libraries = ['protobuf']</code>  
       to <code>libraries = ['libprotobuf', 'libprotobuf-lite']</code>
   *   Change from <code>extra_objects = ['../src/.libs/libprotobuf.a', '../src/.libs/libprotobuf-lite.a']</code>  
       to <code>extra_objects = ['../cmake/build/solution/Release/libprotobuf.lib', '../cmake/build/solution/Release/libprotobuf-lite.lib']</code>
8.  Build the Python package with the C++ implementation:
```shell
python setup.py build –cpp_implementation
```
9.  Install the Python package with the C++ implementation:
```shell
python3 -m easy_install dist/protobuf-3.6.1-py3.6-win-amd64.egg
```
10.  Set an environment variable to boost the protobuf performance:
```shell
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
```

## See Also

* [Converting a Model to Intermediate Representation (IR)](convert_model/Converting_Model.md)
