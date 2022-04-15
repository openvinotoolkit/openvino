# Troubleshooting Steps for OpenVINO™ Installation and Configurations {#openvino_docs_get_started_guide_troubleshooting_steps}

If you run into issues while installing or configuring OpenVINO™, you can try the following methods to do a quick check first. 

## Check the Python and PIP Versions

To check your Python version, run ``python -VV``.

The supported Python versions have the following requirements:

* It should be 64-bit and between 3.6 and 3.9. <!--validation: verify if Windows on 3.6 is still supported-->
* Do not install Python from a Windows Store as it can cause issues. <!--validation: is that still the case for Python 3.8+?-->
* For Ubuntu, use the Python version comes with the system, or install the `libpython3.X` libraries. <!--how to install that? -->
<!--* For macOS, ?-->

For PIP, make sure that your installed version is the latest one. 


## Check the Special Tips for Anaconda Installation

<!--missing part-->


## Check If Required External Dependencies Is Installed

For Ubuntu and RHEL 8 systems, some system libraries are required. <!--what system libraries?-->

If C++ is still required, check if that <!--"that" means Microsoft Visual Studio 2019 with MSBuild and CMake 3.14 or higher, 64-bit?--> is installed and/or point users to it. <!--what does it mean by "point users to it"?-->


## Check If You Have Installed OpenVINO Before 

If you have installed OpenVINO before and added `setupvars` to your `PATH /.bashrc`, do the following steps:

<!--missing part-->


## Check If You Can Import OpenVINO

Use the following coommand to check if you can import OpenVINO:
```sh
python -c "from openvino.runtime import Core"
```

<!--
If yes, ?
or If no, ?
-->

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

If you met proxy issues during the installation with Docker, please set up proxy settings for Docker. See the Proxy section in the [Install the DL Workbench from DockerHub](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Prerequisites.html#set-proxy) topic.
