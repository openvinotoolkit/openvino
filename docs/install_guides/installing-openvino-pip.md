# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip_win}

You can install Intel® Distribution of OpenVINO™ Toolkit Runtime Package through the PyPI repository. 

## System Requirements

#### Below you will find all supported operating systems and required Python\* versions. 

|Operating Systems (64 bit)                        | [Python\* Version](https://www.python.org/downloads/) (64 bit)                 |
|--------------------------------------------------|--------------------------------------------------------------------------------|
|CentOS* 7                                         | 3.6, 3.7, 3.8                                                                  |
|macOS* 10.15.x versions                           | 3.6, 3.7, 3.8                                                                  |
|Red Hat* Enterprise Linux* 8                      | 3.6, 3.8                                                                       |
|Ubuntu* 18.04 long-term support (LTS)             | 3.6, 3.7, 3.8                                                                  |
|Ubuntu* 20.04 long-term support (LTS)             | 3.6, 3.7, 3.8                                                                  |
|Windows 10*                                       | 3.6, 3.7, 3.8                                                                  |

@sphinxdirective

.. note::
   While installing Python, make sure to add it to system PATH.

@endsphinxdirective

## Installing OpenVINO Runtime Package

The OpenVINO Runtime Package contains a set of libraries for an easy inference integration into your applications and supports heterogeneous execution across Intel® CPU and Intel® GPU hardware. To install OpenVINO Runtime Package, use the following procedures:

### 1. Set up Python virtual environment

```
python -m pip install --user virtualenv 
python -m venv openvino_env
```
@sphinxdirective

.. note::

   On Linux and macOS, you may need to type *python3* instead of *python*.

@endsphinxdirective

### 2. Activate virtual environment


@sphinxdirective

.. tabs::

   .. group-tab:: WINDOWS\*

      ```
      openvino_env\Scripts\activate
      ```

   .. group-tab:: Linux\* \| MacOS \*

      ```
      source openvino_env/Scripts/activate
      ```

@endsphinxdirective


### 3. Upgrade PIP

```
python -m pip install --upgrade pip
```

### 4. Install the package

```
pip install openvino
```

### 5. Verify the installation

If installation was successful, you will not see any error messages (no console output) after using the following command.

```
python -c "from openvino.inference_engine import IECore"
```

## Troubleshooting

### You may be prompted with the following errors:

@sphinxdirective

.. tabs::

    .. tab:: WINDOWS\*

      *Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"*
      
      To resolve this issue, you need to install [Build Tools for Visual Studio* 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) and repeat package installation.

    .. tab:: Linux\* \| MacOS \*
      
      *ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory*

      To resolve missing external dependency on Ubuntu* 18.04, execute the following command:

      ```
      sudo apt-get install libpython3.7
      ```

@endsphinxdirective


## Using the OpenVINO Toolkit Runtime Package

You may start with the collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to leverage our API for optimized deep learning inference.

The notebooks can be cloned directly from GitHub. See the [installation guide](https://github.com/openvinotoolkit/openvino_notebooks/wiki/).


## Installing OpenVINO Development Tools

OpenVINO Development Tools include Model Optimizer, Benchmark Tool, Accuracy Checker, Post-Training Optimization Tool and Open Model Zoo tools including Model Downloader. 

While installing OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately.

You will find the information on how to install OpenVINO Development Tools in the following [article](../install_guides/installing-model-dev-tools.md).


## Additional Resources

- [Intel® Distribution of OpenVINO™ Toolkit](https://pypi.org/project/openvino)
- [OpenVINO™ Runtime User Guide](../OV_Runtime_UG/openvino_intro.md)
- [OpenVINO Samples Overview](../OV_Runtime_UG/Samples_Overview.md)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
