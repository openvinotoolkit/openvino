# Install Intel® Distribution of OpenVINO™ toolkit for Windows* from Docker* Image {#openvino_docs_install_guides_installing_openvino_docker_windows}

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The Intel® Distribution of OpenVINO™ toolkit includes the Intel® Deep Learning Deployment Toolkit.  

This guide provides the steps for creating a Docker* image with Intel® Distribution of OpenVINO™ toolkit for Windows* and further installation.

## System Requirements

**Target Operating Systems**

- Windows Server Core*

**Host Operating Systems**

- Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or later) editions
- Windows Server* 2016 or higher 

## Prebuilt Images

Prebuilt images are available on [Docker Hub](https://hub.docker.com/u/openvino).

## Build a Docker* Image for CPU

You can use [available Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your setting via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. 
The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. 

## Install Additional Dependencies
### Install CMake
To add CMake to the image, add the following commands to the Dockerfile:
~~~
RUN powershell.exe -Command `
    Invoke-WebRequest -URI https://cmake.org/files/v3.14/cmake-3.14.7-win64-x64.msi -OutFile %TMP%\\cmake-3.14.7-win64-x64.msi ; `
    Start-Process %TMP%\\cmake-3.14.7-win64-x64.msi -ArgumentList '/quiet /norestart' -Wait ; `
    Remove-Item %TMP%\\cmake-3.14.7-win64-x64.msi -Force

RUN SETX /M PATH "C:\Program Files\CMake\Bin;%PATH%"
~~~
In case of proxy issues, please add the `ARG HTTPS_PROXY` and `-Proxy %%HTTPS_PROXY%` settings to the `powershell.exe` command to the Dockerfile. Then build a docker image:
~~~
docker build . -t <image_name> `
--build-arg HTTPS_PROXY=<https://your_proxy_server:port>
~~~

### Install Microsoft Visual Studio* Build Tools
You can add Microsoft Visual Studio Build Tools* to a Windows* OS Docker image. Available options are to use offline installer for Build Tools 
(follow the [Instruction for the offline installer](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019)) or 
to use the online installer for Build Tools (follow [Instruction for the online installer](https://docs.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2019)).
Microsoft Visual Studio Build Tools* are licensed as a supplement your existing Microsoft Visual Studio* license. 
Any images built with these tools should be for your personal use or for use in your organization in accordance with your existing Visual Studio* and Windows* licenses.

To add MSBuild 2019 to the image, add the following commands to the Dockerfile:
~~~
RUN powershell.exe -Command Invoke-WebRequest -URI https://aka.ms/vs/16/release/vs_buildtools.exe -OutFile %TMP%\\vs_buildtools.exe

RUN %TMP%\\vs_buildtools.exe --quiet --norestart --wait --nocache `
	 --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools" `
     --add Microsoft.VisualStudio.Workload.MSBuildTools `
     --add Microsoft.VisualStudio.Workload.UniversalBuildTools `
     --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended `
     --remove Microsoft.VisualStudio.Component.Windows10SDK.10240 `
     --remove Microsoft.VisualStudio.Component.Windows10SDK.10586 `
     --remove Microsoft.VisualStudio.Component.Windows10SDK.14393 `
     --remove Microsoft.VisualStudio.Component.Windows81SDK || IF "%ERRORLEVEL%"=="3010" EXIT 0 && powershell set-executionpolicy remotesigned
~~~
In case of proxy issues, please use an offline installer for Build Tools (follow [Instruction for the offline installer](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019).

## Run the Docker* Image for CPU

To install the OpenVINO toolkit from the prepared Docker image, run the image with the following command (currently support only CPU target):
~~~
docker run -it --rm <image_name>
~~~

If you want to try some demos then run image with the root privileges (some additional 3-rd party dependencies will be installed):
~~~
docker run -itu ContainerAdministrator --rm <image_name> cmd /S /C "cd deployment_tools\demo && demo_security_barrier_camera.bat -d CPU -sample-options -no_show"
~~~

## Troubleshooting

If you got proxy issues, please setup proxy settings for Docker. See the Proxy section in the [Install the DL Workbench from Docker Hub* ](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) topic.

## Additional Resources

* [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.

* Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)  

* OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)

* Intel® Distribution of OpenVINO™ toolkit Docker Hub* home page: [https://hub.docker.com/u/openvino](https://hub.docker.com/u/openvino)
