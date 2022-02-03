# Build Instructions {#ovtf_build}

@sphinxdirective
1. `Prerequisites`_

   * `Ubuntu`_
   * `macOS`_
   * `Windows`_

2. `OpenVINO™ integration with TensorFlow`_

   * `Build Instructions for Linux and macOS`_
   * `Build Instructions for Windows`_
   * `Build Instructions for Intel Atom® Processor`_
   * `Build Verification`_

3. `OpenVINO™`_
4. `TensorFlow`_
5. `Build ManyLinux2014 compatible OpenVINO™ integration with TensorFlow wheels`_
@endsphinxdirective


@sphinxdirective
.. _Prerequisites:
@endsphinxdirective

## 1. Prerequisites

The first step is cloning the `openvino_tensorflow` repository:

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```


@sphinxdirective
.. _Ubuntu:
@endsphinxdirective

### Ubuntu

1. Install apt packages  

      ```bash  
      # users may need sudo access to install some of the dependencies  
      $ apt-get update
      $ apt-get install python3 python3-dev
      $ apt-get install -y --no-install-recommends ca-certificates autoconf automake build-essential \
      libtool unzip git unzip wget zlib1g zlib1g-dev bash-completion \
      build-essential cmake zip golang-go locate curl clang-format cpio libtinfo-dev jq \
      lsb-core libusb-1.0-0-dev patchelf
    
      # install the required gcc package (Optional; if gcc-7 is not installed)
      $ apt-get install gcc-7 g++-7  

      # if multiple gcc versions are installed then install alternatives
      $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 
      $ update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

      # configure alternatives (select the required gcc version)
      $ update-alternatives --config gcc
      $ update-alternatives --config g++
      ```        

2. Install Python requirements  

      ```bash
      $ pip3 install -r requirements.txt
      ```  

3. Install CMake, supported version >=3.14.0 and =<3.20.0, 
        
      ```bash
      # to install CMake 3.18.2
      $ wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ tar -xzvf cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ cp cmake-3.18.4-Linux-x86_64/bin/* /usr/local/bin/ && \
        cp -r cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/local/share/
      ```

4. Install Bazelisk (Optional; required only while building TensorFlow from source)

      ```bash
      $ apt-get update && apt-get install -y openjdk-8-jdk
      $ curl -fsSL https://deb.nodesource.com/setup_12.x | bash -
      $ apt-get install -y nodejs
      $ npm install -g @bazel/bazelisk
      ```
@sphinxdirective
.. _macOS:
@endsphinxdirective

### macOS

1. Install HomeBrew packages

      ``` bash
      $ brew install python@3.9
      $ brew install cmake autoconf automake libtool libusb wget 
      ```

2. Install Pip and requirements

      ```bash
      $ wget https://bootstrap.pypa.io/get-pip.py
      $ python3 get-pip.py
      $ pip3 install -r requirements.txt
      ```

3. Install Bazelisk v1.10.1 (Optional; required only while building TensorFlow from source)

      ```bash
      $ wget https://github.com/bazelbuild/bazelisk/releases/download/v1.10.1/bazelisk-darwin-amd64
      $ mv bazelisk-darwin-amd64 /usr/local/bin/bazel
      $ chmod 777 /usr/local/bin/bazel
      ```

4. Install Apple XCode Command Line Tools

      ```bash
      $ xcode-select --install
      ```

**Note:**  
Developed and Tested on macOS version 11.2.3 with CMake version 3.21.1  
User can install Python 3.7, 3.8, or 3.9. 


@sphinxdirective
.. _Windows:
@endsphinxdirective

### Windows
1. Install [Python3.9](https://www.python.org/downloads/)

2. Install Visual C++ Build Tools 2019   
   Visual C++ build tools 2019 comes with Visual Studio 2019 but can be installed separately:
   - Go to the Visual Studio [downloads](https://visualstudio.microsoft.com/downloads/),  
   - Select Redistributables and Build Tools,
   - Download and install:  
      Microsoft Visual C++ 2019 Redistributable  
      Microsoft Build Tools 2019

3. Install [CMake](https://cmake.org/download/), supported version >=3.14.0 and =<3.20.1,
4. [Download and Install OpenVINO™ Toolkit 2021.4.2 LTS for Windows](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html)

@sphinxdirective
.. _OpenVINO™ integration with TensorFlow:
@endsphinxdirective

## 2. OpenVINO™ integration with TensorFlow

@sphinxdirective
.. _Build Instructions for Linux and macOS:
@endsphinxdirective

### Build Instructions for Linux and macOS

Use one of the following build options based on the requirements. **OpenVINO™ integration with TensorFlow** built using PyPI TensorFlow enables only the Python APIs. TensorFlow C++ libraries built from source is required to use the C++ APIs.

To enable C++ APIs, build **OpenVINO™ integration with TensorFlow** with TensorFlow C++ libraries (using the option --use_tensorflow_from_location). For VAD-M support, build with OpenVINO™ binary (using the option --use_openvino_from_location).

1. Pulls compatible TensorFlow package from PyPi, clones and builds OpenVINO™ from source. The arguments are optional. If any argument is not provided, then the default versions as specified in build_ovtf.py will be used. This is the recommended build option for most of the use cases. 

        python3 build_ovtf.py --tf_version=v2.7.0 --openvino_version=2021.4.2

2. Pulls compatible TensorFlow ABI1 package from Github release assets. Uses OpenVINO™ binary from specified location.

        python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

3. Uses pre-built TensorFlow from the given location ([refer the TensorFlow build instructions](#LinuxAndmacOS)). Pulls and builds OpenVINO™ from source. Use this if you need to build **OpenVINO™ integration with TensorFlow** frequently without building TensorFlow from source everytime.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/

4. Uses prebuilt TensorFlow from the given location ([refer the TensorFlow build instructions](#LinuxAndmacOS)). Uses OpenVINO™ binary from specified location. **This is only compatible with ABI1 built TensorFlow**.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/ --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

Select the `help` option of `build_ovtf.py` script to learn more about various build options.

        python3 build_ovtf.py --help


@sphinxdirective
.. _Build Instructions for Windows:
@endsphinxdirective

### Build Instructions for Windows

1. Pulls compatible prebuilt TensorFlow package from Github release assets. Uses OpenVINO™ binary from specified location.

        python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2021.4.752"

2. Uses prebuilt TensorFlow from the given location ([refer to the TensorFlow build instructions](#TFWindows). Uses OpenVINO™ binary from specified location.

        python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2021.4.752" --use_tensorflow_from_location="\path\to\directory\containing\tensorflow\"


@sphinxdirective
.. _Build Instructions for Intel Atom® Processor:
@endsphinxdirective

### Build Instructions for Intel Atom® Processor

In order to build **OpenVINO™ integration with TensorFlow** to use with the Intel Atom® processor, we recommend building TensorFlow from source, by using the following command:

        python3 build_ovtf.py --build_tf_from_source --cxx11_abi_version=1 --target_arch silvermont

@sphinxdirective
.. _Build Verification:
@endsphinxdirective

### Build Verification

When the build is finished, a new `virtualenv` directory with the name of `venv-tf-py3` is created in `build_cmake`. Build artifacts (e.g. the `openvino_tensorflow-<VERSION>-cp38-cp38m-manylinux2014_x86_64.whl`) are created in the `build_cmake/artifacts/` directory.

Activate the following `virtualenv` to start using **OpenVINO™ integration with TensorFlow**.

        source build_cmake/venv-tf-py3/bin/activate

Alternatively, you may install the TensorFlow and **OpenVINO™ integration with TensorFlow** outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Verify that `openvino-tensorflow` is installed correctly:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.7.0
        OpenVINO integration with TensorFlow version: b'1.1.0'
        OpenVINO version used for this build: b'2021.4.2'
        TensorFlow version used for this build: v2.7.0
        CXX11_ABI flag used for this build: 1


Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO™.

@sphinxdirective 
 .. _OpenVINO™:
@endsphinxdirective

## 3. OpenVINO™

OpenVINO™ can be built from source independently using `build_ov.py`

@sphinxdirective
.. _TensorFlow:
@endsphinxdirective

## 4. TensorFlow

@sphinxdirective
.. _LinuxAndmacOS:
@endsphinxdirective

### 4. For Linux and macOS

TensorFlow can be built from source using `build_tf.py`. The build artifacts can be found under ${PATH_TO_TF_BUILD}/artifacts/

- Set your build path

        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- For all available build options

        python3 build_tf.py -h

- Builds TensorFlow with CXX11_ABI=0 with a desired TensorFlow version

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.7.0 --cxx11_abi_version=0 

- Builds TensorFlow with CXX11_ABI=1 with a desired TensorFlow version

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.7.0 --cxx11_abi_version=1

@sphinxdirective
.. _TFWindows:
@endsphinxdirective

### For Windows

- Complete the setup steps: https://www.tensorflow.org/install/source_windows#setup_for_windows
- Download TensorFlow source code and apply patch
  
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v2.7.0
        # apply following patch to enable the symbols required to build OpenVINO™ integration with TensorFlow
        git apply patch \path\to\openvino_tensorflow\repo\tools\builds\tf_2_7_0_build.patch
        # if you want to enable more symbols add them to tensorflow\tensorflow\tools\def_file_filter\def_file_filter.py.tpl file

- Configure the Build: https://www.tensorflow.org/install/source_windows#configure_the_build      
- Build the PIP Package

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

        # following command builds a .whl package in the C:/tmp/tensorflow_pkg directory:  
        bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg

        # copy this PIP package to working directory (tensorflow source code directory)
        cp C:/tmp/tensorflow_pkg/tensorflow-2.7.0-cp39-cp39-win_amd64.whl .\
  
- Build CC libraries
  
        bazel build --config=opt tensorflow:tensorflow_cc.lib


@sphinxdirective
.. _Build ManyLinux2014 compatible OpenVINO™ integration with TensorFlow wheels:
@endsphinxdirective

## 5. Build ManyLinux2014 compatible **OpenVINO™ integration with TensorFlow** wheels

To build wheel files compatible with manylinux2014, use the following commands. The build artifacts will be available in your container's /whl/ folder.

```bash
cd tools/builds/
docker build --no-cache -t openvino_tensorflow/pip --build-arg OVTF_BRANCH=releases/v1.1.0 . -f Dockerfile.manylinux2014
```
