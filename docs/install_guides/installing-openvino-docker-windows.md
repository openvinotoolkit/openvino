# Install Intel® Distribution of OpenVINO™ toolkit for Windows* from Docker* Image {#openvino_docs_install_guides_installing_openvino_docker_windows}

This guide provides steps for creating a Docker* image with Intel® Distribution of OpenVINO™ toolkit for Windows* and using the Docker image on different devices.

## System requirements

**Target Operating Systems**

- Windows Server Core*

**Host Operating Systems**

- Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or later) editions
- Windows Server* 2016 or higher

## Creating an OpenVINO Docker image

You can create a Docker image with OpenVINO via either of the following ways. 

### Get a prebuilt image directly from provided sources

You can find prebuilt images on [Docker Hub](https://hub.docker.com/u/openvino).

### Build a Docker* image manually

You can use the [available Dockerfiles on GitHub](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your setting via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci)which can generate a Dockerfile, build, test, and deploy an image with the the Intel® Distribution of OpenVINO™ toolkit.

## Using OpenVINO Docker image on different devices

### Using OpenVINO Docker* image on CPU

#### Step 1: Install additional dependencies

- **Install CMake**

   To add CMake to the image, add the following commands to the Dockerfile:
   ```bat
   RUN powershell.exe -Command `
       Invoke-WebRequest -URI https://cmake.org/files/v3.14/cmake-3.14.7-win64-x64.msi -OutFile %TMP%\\cmake-3.14.7-win64-x64.msi ; `
       Start-Process %TMP%\\cmake-3.14.7-win64-x64.msi -ArgumentList '/quiet /norestart' -Wait ; `
       Remove-Item %TMP%\\cmake-3.14.7-win64-x64.msi -Force

   RUN SETX /M PATH "C:\Program Files\CMake\Bin;%PATH%"
   ```

   In case of proxy issues, please add the `ARG HTTPS_PROXY` and `-Proxy %%HTTPS_PROXY%` settings to the `powershell.exe` command to the Dockerfile. Then build a Docker image:
   ```bat
   docker build . -t <image_name> `
   --build-arg HTTPS_PROXY=<https://your_proxy_server:port>
   ```   
   
- **Install Microsoft Visual Studio\* Build Tools**

   You can add Microsoft Visual Studio Build Tools* to a Windows* OS Docker image using the [offline](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019) or [online](https://docs.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2019) installers for Build Tools.
   
   Microsoft Visual Studio Build Tools* are licensed as a supplement your existing Microsoft Visual Studio* license.
   
   Any images built with these tools should be for your personal use or for use in your organization in accordance with your existing Visual Studio* and Windows* licenses.

   To add MSBuild 2019 to the image, add the following commands to the Dockerfile:
   ```bat
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
   ```

   In case of proxy issues, please use the [offline installer for Build Tools](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019).
   

#### Step 2: Run OpenVINO Docker* image on CPU

To start the interactive session, run the following command allows inference on the CPU:

```bat
docker run -it --rm <image_name>
```

If you want to try some demos then run image with the root privileges (some additional 3-rd party dependencies will be installed):

```bat
docker run -it --rm <image_name> 
cmd /S /C "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp CPU"
```

### Using OpenVINO Docker* image on GPU

GPU Acceleration in Windows containers feature requires to meet Windows host, OpenVINO toolkit and Docker* requirements:

- [Windows requirements](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/gpu-acceleration):
  - The container host must be running Windows Server 2019 or Windows 10 of version 1809 or higher.
  - The container base image must be `mcr.microsoft.com/windows:1809` or higher. Windows Server Core and Nano Server container images are not currently supported.
  - The container host must be running Docker Engine 19.03 or higher.
  - The container host must have GPU running display drivers of version WDDM 2.5 or higher.
- [OpenVINO™ GPU requirement](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html#Install-GPU):
  - Intel Graphics Driver for Windows of version 15.65 or higher.
- [Docker isolation mode requirement](https://docs.microsoft.com/en-us/virtualization/windowscontainers/manage-containers/hyperv-container):
  - Windows host and container version tags must match.
  - [Windows host and container isolation process support](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility)

#### Step 1: Configure OpenVINO Docker* image for your host system

1. Reuse one of [available Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles). You can also use your own Dockerfile. 
2. Check your [Windows host and container isolation process compatibility](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility).
3. Find the appropriate Windows container base image on [DockerHub*](https://hub.docker.com/_/microsoft-windows) and set up your host/container version in the `FROM` Dockerfile instruction.  
   For example, in the `openvino_c_dev_<version>.dockerfile`, change:  
   ```bat
   FROM mcr.microsoft.com/windows/servercore:ltsc2019 AS ov_base
   ```
   to:
   ```bat
   FROM mcr.microsoft.com/windows:20H2
   ```
4. Build the Docker image by running the following command:
   ```bat
   docker build --build-arg package_url=<OpenVINO pkg> -f <Dockerfile> -t <image_name> .
   ```
5. Copy `OpenCL.dll` from your `C:\Windows\System32` host folder to any `temp` directory:
   ```bat
   mkdir C:\tmp
   copy C:\Windows\System32\OpenCL.dll C:\tmp
   ```

#### Step 2: Run OpenVINO Docker* image on GPU

1. To try inference on a GPU, run the image with the following command:
   ```bat
   docker run -it --rm -u ContainerAdministrator --isolation process --device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599 -v C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409:C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409 -v C:\tmp:C:\tmp <image_name>
   ```
   where
   - `--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599` is a reserved interface class GUID for a GPU device.
   - `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409` is the path to OpenCL driver home directory. To find it on your PC, run the `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*` regular expression.
   - `C:\tmp` is the folder with the copy of `OpenCL.dll` from your `C:\Windows\System32` host folder.
2. Copy `OpenCL.dll` to the `C:\Windows\System32` folder inside the container and set appropriate registry entry. Now you can run inference on a GPU device:
   ```bat
   copy C:\tmp\OpenCL.dll C:\Windows\System32\ && reg add "HKLM\SOFTWARE\Khronos\OpenCL\Vendors" /v "C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409\ocl\bin\x64\intelocl64.dll" /t REG_DWORD /d 0
   ```
   For example, run the `Hello Classification Python` sample with the following command:
   ```bat
   omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp GPU
   ```
   > **NOTE**: Additional third-party dependencies will be installed.


## Additional resources

- [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
