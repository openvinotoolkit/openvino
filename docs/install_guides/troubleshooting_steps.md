# Troubleshooting Steps for OpenVINO™ Installation and Configurations {#openvino_docs_get_started_guide_troubleshooting_steps}

If you run into issues while installing or configuring OpenVINO™, you can try the following methods to do some quick checks first. 

## Check the Python and PIP versions

To check your Python version, run `python -VV` or `python --version`. The supported Python versions should be 64-bit and between 3.7 and 3.9. Note that Python 3.6 is not supported anymore.

If your Python version does not meet the requirements, update Python:

* For Windows, **do not install Python from a Windows Store** as it can cause issues. You are highly recommended to install Python from <https://www.python.org/>.
* For Ubuntu and other Linux systems, use the Python version comes with the system, or install the `libpython3.X` libraries via the following commands (taking Python 3.7 as an example):
```sh
sudo apt-get install libpython3.7
sudo apt-get install libpython3-dev
```
* For macOS, download a proper Python version from <https://www.python.org/> and install it. Note that macOS 10.x comes with python 2.7 installed, which is not supported, so you still need install Python from its official website.

For PIP, make sure that you have installed the latest version. To check and upgrade your PIP version, run the following command:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      pip install --upgrade pip
   
.. tab:: Windows

   .. code-block:: sh
   
      python -m pip install --upgrade pip


@endsphinxdirective


## Check the special tips for Anaconda installation

<!--missing part-->


## Check if required external dependencies is installed

For Ubuntu and RHEL 8 systems, if you installed OpenVINO Runtime via the installer, APT, or YUM repository, and decided to [install OpenVINO Development Tools](installing-model-dev-tools.md), make sure that you <a href="openvino_docs_install_guides_installing_openvino_linux.html#install-external-dependencies">Install External Software Dependencies</a> first. 

 <!--if you installed OpenVINO via PyPI, is it necessary to get system libraries or install dependencies? and how? Does the [install OpenVINO Development Tools](installing-model-dev-tools.md) page have enough information?-->

For Windows systems, if C++ is still required, make sure that Microsoft Visual Studio 2019 with MSBuild and CMake 3.14 or higher (64-bit) are installed. While installing Microsoft Visual Studio 2019, make sure that you have selected **Desktop development with C++** in the **Workloads** tab. If not, launch the installer again to select that option. For more information on modifying the installation options for Microsoft Visual Studio, see its [official support page](https://docs.microsoft.com/en-us/visualstudio/install/modify-visual-studio?view=vs-2022).

## Check if you have installed OpenVINO before 

If you have installed OpenVINO before and added `setupvars` to your `PATH /.bashrc`, do the following steps:

<!--missing part-->


## Verify if OpenVINO is correctly installed

To verify if OpenVINO is correctly installed, use the following command:
```sh
python -c "from openvino.runtime import Core"
```

If OpenVINO was successfully installed, nothing will happen. If not, an error will be displayed. 

## Check the Version of OpenVINO Runtime and Developement Tools

<!--missing part
Which commands/steps are needed to show the version?
And what should the user do if the version is different?
And is this for both 21.4 and 22.x?
-->

## Firewall/Network Issues

<!--the following content is taken from troubleshooting.md. Please check if this is what you mean.-->

### Errors with Installing via PIP for PRC Users

Users in People's Repulic of China (PRC) might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try one of the following options:
   
* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   ``` sh
   pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install specific framework. For example:
   
   ```
   pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
   ```
   
* If you run into incompatibility issues between components after installing OpenVINO, try running ``requirements.txt`` with the following command:

   ``` sh
   pip install -r <INSTALL_DIR>/tools/requirements.txt
   ```

### Proxy Issues with Installing OpenVINO on Linux from Docker

If you met proxy issues during the installation with Docker, set up the proxy settings for Docker. See the Proxy section in [Install the DL Workbench from DockerHub](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Prerequisites.html#set-proxy).


## Check if GPU drvier is installed

[Additional configurations](configurations-header.md) are required in order to use OpenVINO on different hardware.

To run inference on GPU, make sure that you have installed the correct GPU driver. To check that, see [additional configurations for GPU](configurations-for-intel-gpu.md).