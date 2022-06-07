# Install Intel® Distribution of OpenVINO™ toolkit from Anaconda Cloud {#openvino_docs_install_guides_installing_openvino_conda}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit through the Anaconda Cloud.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

## System Requirements

@sphinxdirective

.. tab:: Anaconda Software

   [Anaconda distribution](https://www.anaconda.com/products/individual/)

.. tab:: Supported Operating Systems with Python Version

   +--------------------------------------+----------------------------------------------------+
   | Operating System (64-bit)            | [Python Version (64-bit)](https://www.python.org/) |
   +======================================+====================================================+
   | macOS 10.15                          | 3.6, 3.7, 3.8, 3.9                                 |
   +--------------------------------------+----------------------------------------------------+
   | Red Hat Enterprise Linux 8           | 3.6, 3.7, 3.8, 3.9                                 |
   +--------------------------------------+----------------------------------------------------+
   | Ubuntu 18.04 long-term support (LTS) | 3.6, 3.7, 3.8, 3.9                                 |
   +--------------------------------------+----------------------------------------------------+
   | Ubuntu 20.04 long-term support (LTS) | 3.6, 3.7, 3.8, 3.9                                 |
   +--------------------------------------+----------------------------------------------------+
   | Windows 10                           | 3.6, 3.7, 3.8, 3.9                                 |
   +--------------------------------------+----------------------------------------------------+  


@endsphinxdirective

## Install OpenVINO Runtime Using the Anaconda Package Manager

1. Set up the Anaconda environment (taking Python 3.7 for example): 
   ```sh
   conda create --name py37 python=3.7
   conda activate py37
   ```
2. Update Anaconda environment to the latest version:
   ```sh
   conda update --all
   ```
3. Install the Intel® Distribution of OpenVINO™ toolkit:

@sphinxdirective

.. tab:: Red Hat Enterprise Linux 8

   .. code-block:: sh

      conda install openvino-ie4py-rhel8 -c intel

.. tab:: Ubuntu 18.04

   .. code-block:: sh

      conda install openvino-ie4py-ubuntu18 -c intel

.. tab:: Ubuntu 20.04

   .. code-block:: sh

      conda install openvino-ie4py-ubuntu20 -c intel

.. tab:: Windows 10 and macOS

   .. code-block:: sh

     conda install openvino-ie4py -c intel


@endsphinxdirective

4. Verify if the package is successfully installed:
   ```sh
   python -c "from openvino.runtime import Core"
   ```
   If installation was successful, there will not be any error messages (no console output) present.

5. Now, you may start developing your application.

## What's Next?

Now, you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/latest/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
- Intel® Distribution of OpenVINO™ toolkit Anaconda home page: [https://anaconda.org/intel/openvino-ie4py](https://anaconda.org/intel/openvino-ie4py)