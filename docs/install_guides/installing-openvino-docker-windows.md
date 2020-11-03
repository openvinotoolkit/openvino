# Install Intel® Distribution of OpenVINO™ toolkit for Windows* from Docker* Image {#openvino_docs_install_guides_installing_openvino_docker_windows}

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The Intel® Distribution of OpenVINO™ toolkit includes the Intel® Deep Learning Deployment Toolkit.  

This guide provides the steps for creating a Docker* image with Intel® Distribution of OpenVINO™ toolkit for Windows* and further installation.

## System Requirements

**Target Operating Systems**

- Windows Server Core*

**Host Operating Systems**

- Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or later) editions
- Windows Server* 2016 or higher 

## Build a Docker* Image for CPU

To build a Docker image, create a `Dockerfile` that contains defined variables and commands required to create an OpenVINO toolkit installation image. 

Create your `Dockerfile` using the following example as a template:

<details>
  <summary>Click to expand/collapse</summary>

~~~
# escape= `
FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

USER ContainerAdministrator

# Setup Redistributable Libraries for Intel(R) C++ Compiler for Windows*

RUN powershell.exe -Command `
    Invoke-WebRequest -URI https://software.intel.com/sites/default/files/managed/59/aa/ww_icl_redist_msi_2018.3.210.zip -Proxy %HTTPS_PROXY%  -OutFile "%TMP%\ww_icl_redist_msi_2018.3.210.zip" ; `
    Expand-Archive -Path "%TMP%\ww_icl_redist_msi_2018.3.210.zip" -DestinationPath "%TMP%\ww_icl_redist_msi_2018.3.210" -Force ; `
    Remove-Item "%TMP%\ww_icl_redist_msi_2018.3.210.zip" -Force

RUN %TMP%\ww_icl_redist_msi_2018.3.210\ww_icl_redist_intel64_2018.3.210.msi /quiet /passive /log "%TMP%\redist.log"

# setup Python
ARG PYTHON_VER=python3.7

RUN powershell.exe -Command `
  Invoke-WebRequest -URI https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe -Proxy %HTTPS_PROXY% -OutFile %TMP%\\python-3.7.exe ; `
  Start-Process %TMP%\\python-3.7.exe -ArgumentList '/passive InstallAllUsers=1 PrependPath=1 TargetDir=c:\\Python37' -Wait ; `
  Remove-Item %TMP%\\python-3.7.exe -Force

RUN python -m pip install --upgrade pip
RUN python -m pip install cmake

# download package from external URL
ARG package_url=http://registrationcenter-download.intel.com/akdlm/irc_nas/16613/w_openvino_toolkit_p_0000.0.000.exe
ARG TEMP_DIR=/temp

WORKDIR ${TEMP_DIR}
ADD ${package_url} ${TEMP_DIR}

# install product by installation script
ARG build_id=0000.0.000
ENV INTEL_OPENVINO_DIR C:\intel

RUN powershell.exe -Command `
    Start-Process "./*.exe" -ArgumentList '--s --a install --eula=accept --installdir=%INTEL_OPENVINO_DIR% --output=%TMP%\openvino_install_out.log --components=OPENVINO_COMMON,INFERENCE_ENGINE,INFERENCE_ENGINE_SDK,INFERENCE_ENGINE_SAMPLES,OMZ_TOOLS,POT,INFERENCE_ENGINE_CPU,INFERENCE_ENGINE_GPU,MODEL_OPTIMIZER,OMZ_DEV,OPENCV_PYTHON,OPENCV_RUNTIME,OPENCV,DOCS,SETUPVARS,VC_REDIST_2017_X64,icl_redist' -Wait

ENV INTEL_OPENVINO_DIR C:\intel\openvino_${build_id}

# Post-installation cleanup
RUN rmdir /S /Q "%USERPROFILE%\Downloads\Intel"

# dev package
WORKDIR ${INTEL_OPENVINO_DIR}
RUN python -m pip install --no-cache-dir setuptools && `
    python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\python\%PYTHON_VER%\requirements.txt" && `
    python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\python\%PYTHON_VER%\openvino\tools\benchmark\requirements.txt" && `
    python -m pip install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR ${TEMP_DIR}
COPY scripts\install_requirements.bat install_requirements.bat
RUN install_requirements.bat %INTEL_OPENVINO_DIR%


WORKDIR ${INTEL_OPENVINO_DIR}\deployment_tools\open_model_zoo\tools\accuracy_checker
RUN %INTEL_OPENVINO_DIR%\bin\setupvars.bat && `
    python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker\requirements.in" && `
    python "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker\setup.py" install

WORKDIR ${INTEL_OPENVINO_DIR}\deployment_tools\tools\post_training_optimization_toolkit
RUN python -m pip install --no-cache-dir -r "%INTEL_OPENVINO_DIR%\deployment_tools\tools\post_training_optimization_toolkit\requirements.txt" && `
    python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\post_training_optimization_toolkit\setup.py" install

WORKDIR ${INTEL_OPENVINO_DIR}

# Post-installation cleanup
RUN powershell Remove-Item -Force -Recurse "%TEMP%\*" && `
    powershell Remove-Item -Force -Recurse "%TEMP_DIR%" && `
    rmdir /S /Q "%ProgramData%\Package Cache"

USER ContainerUser

CMD ["cmd.exe"]
~~~

</details>

> **NOTE**: Replace direct link to the Intel® Distribution of OpenVINO™ toolkit package to the latest version in the `package_url` variable and modify install package name in the subsequent commands. You can copy the link from the [Intel® Distribution of OpenVINO™ toolkit download page](https://software.seek.intel.com/openvino-toolkit) after registration. Right click the **Offline Installer** button on the download page for Linux in your browser and press **Copy link address**.
> **NOTE**: Replace build number of the package in the `build_id` variable according to the name of the downloaded Intel® Distribution of OpenVINO™ toolkit package. For example, for the installation file `w_openvino_toolkit_p_2020.3.333.exe`, the `build_id` variable should have the value `2020.3.333`.

To build a Docker* image for CPU, run the following command:
~~~
docker build . -t <image_name> `
--build-arg HTTP_PROXY=<http://your_proxy_server.com:port> `
--build-arg HTTPS_PROXY=<https://your_proxy_server.com:port>
~~~

## Install additional dependencies
### Install CMake
To add CMake to the image, add the following commands to the `Dockerfile` example above:
~~~
RUN powershell.exe -Command `
    Invoke-WebRequest -URI https://cmake.org/files/v3.14/cmake-3.14.7-win64-x64.msi -Proxy %HTTPS_PROXY% -OutFile %TMP%\\cmake-3.14.7-win64-x64.msi ; `
    Start-Process %TMP%\\cmake-3.14.7-win64-x64.msi -ArgumentList '/quiet /norestart' -Wait ; `
    Remove-Item %TMP%\\cmake-3.14.7-win64-x64.msi -Force

RUN SETX /M PATH "C:\Program Files\CMake\Bin;%PATH%"
~~~

### Install Microsoft Visual Studio* Build Tools
You can add Microsoft Visual Studio Build Tools* to Windows* OS Docker image. Available options are to use offline installer for Build Tools 
(follow [Instruction for the offline installer](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019) or 
to use online installer for Build Tools (follow [Instruction for the online installer](https://docs.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2019).
Microsoft Visual Studio Build Tools* are licensed as a supplement your existing Microsoft Visual Studio* license. 
Any images built with these tools should be for your personal use or for use in your organization in accordance with your existing Visual Studio* and Windows* licenses.

## Run the Docker* Image for CPU

To install the OpenVINO toolkit from the prepared Docker image, run the image with the following command:
~~~
docker run -it <image_name>
~~~

## Additional Resources

* [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.

* Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)  

* OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)

* Intel® Distribution of OpenVINO™ toolkit Docker Hub* home page: [https://hub.docker.com/u/openvino](https://hub.docker.com/u/openvino)
