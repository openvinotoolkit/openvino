# Issues & Solutions for OpenVINO™ Installation & Configuration {#openvino_docs_get_started_guide_troubleshooting_issues}

This page lists issues that you may encounter during the installation and configuration of OpenVINO™, as well as their possible solutions.

## <a name="install-for-prc"></a>Errors with Installing via PIP for Users in China

Users in China might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try one of the following options:
   
* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   ``` sh
   pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install specific framework. For example:
   
   ```
   pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
   ```

* For C++ developers, if you have installed OpenVINO Runtime via APT, YUM, or the archive file, and then installed OpenVINO Development Tools via PyPI, you may run into issues. To resolve that, install the components in ``requirements.txt`` by using the following command: 
   ``` sh
   pip install -r <INSTALL_DIR>/tools/requirements.txt
   ```
  For APT and YUM users, replace the `INSTALL_DIR` with `/usr/share/openvino`.

<!-- this part was from Docker installation -->

## Issues with Installing OpenVINO on Linux from Docker

### <a name="proxy-issues"></a>Proxy Issues

If you met proxy issues during the installation with Docker, you need set up proxy settings for Docker. See the [Set Proxy section in DL Workbench Installation](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Prerequisites.html#set-proxy) for more details.


@anchor yocto-install-issues
## Issues with Creating a Yocto Image for OpenVINO

### Error while adding "meta-intel" layer

When using the `bitbake-layers add-layer meta-intel` command, the following error might occur:
```sh
NOTE: Starting bitbake server...
ERROR: The following required tools (as specified by HOSTTOOLS) appear to be unavailable in PATH, please install them in order to proceed: chrpath diffstat pzstd zstd
```

To resolve the issue, install the `chrpath diffstat zstd` tools:

```sh
sudo apt-get install chrpath diffstat zstd
```

