# Install Intel® Distribution of OpenVINO™ toolkit for Windows from Docker Image {#openvino_docs_install_guides_installing_openvino_docker_windows}

This guide provides instruction for creating a Docker image with Intel® Distribution of OpenVINO™ toolkit for Windows, and using the Docker image on different devices.

## <a name="system-requirments"></a>System Requirements

@sphinxdirective

.. tab:: Supported Operating Systems with Python Version

  +------------------------------------+--------------------------+
  | Target Operating System            | Python Version           |
  +====================================+==========================+
  | Windows Server Core base LTSC 2019 | 3.8                      |
  +------------------------------------+--------------------------+
  | Windows 10, version 20H2           | 3.8                      |
  +------------------------------------+--------------------------+

.. tab:: Host Operating Systems

   * Windows 10 (64-bit) Pro, Enterprise, Education (1607 Anniversary Update, Build 14393 or later),
   * Windows Server 2016 or higher.


@endsphinxdirective

### Additional Requirements for GPU

To use GPU Acceleration in Windows containers, make sure that the following requirements for Windows host, OpenVINO and Docker are met:

- [Windows requirements](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/gpu-acceleration):
   - The container host must be running Windows Server 2019 or Windows 10 of version 1809 or higher.
   - The container base image must be `mcr.microsoft.com/windows:1809` or higher. Windows Server Core and Nano Server container images are not currently supported.
   - The container host must be running Docker Engine 19.03 or higher.
   - The container host must have GPU running display drivers of version WDDM 2.5 or higher.
- GPU requirement for OpenVINO: 
   - Intel Graphics Driver for Windows of version 15.65 or higher.
- [Docker isolation mode requirements](https://docs.microsoft.com/en-us/virtualization/windowscontainers/manage-containers/hyperv-container):
   - Windows host and container version tags must match.
   - [Windows host and container isolation process support](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility).

## Installation Flow

There are two ways to install OpenVINO with Docker. You may choose either of them according to your needs:

### Using a prebuilt image

**Step 1**: Get a prebuilt image from provided sources.

Prebuilt images can be find on:

- [Docker Hub](https://hub.docker.com/u/openvino),
- [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino).

**Step 2**: Run the image on different devices.

@sphinxdirective

.. tab:: Running the Image on CPU

   To start the interactive session, run the following command:

   .. code-block:: sh

      docker run -it --rm <image_name>

   If you want to try some samples, run the image with the following command:

   .. code-block:: sh
   
      docker run -it --rm <image_name>
      cmd /S /C "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp CPU"

.. tab:: Running the Image on GPU

   .. note::
      Since GPU is not supported in prebuilt images or [default Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles), you must make sure the Additional Requirements for GPU in <a href="#system-requirements">System Requirements</a> are met, and configure and build the image manually before you may run inferences on a GPU.

   1. To try inference on a GPU, run the image with the following command:

   .. code-block:: sh
      
      docker run -it --rm -u ContainerAdministrator --isolation process --device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599 -v C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409:C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409 -v C:\tmp:C:\tmp <image_name>
   
   where:
   - `--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599` is a reserved interface class GUID for a GPU device.
   - `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409` is the path to OpenCL driver home directory. To find it on your PC, run the `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*` regular expression.
   - `C:\tmp` is the folder with the copy of `OpenCL.dll` from your `C:\Windows\System32` host folder.
   
   2. Copy `OpenCL.dll` to the `C:\Windows\System32` folder inside the container and set appropriate registry entry. Now you may run inference on a GPU device:
      
   .. code-block:: sh
   
      copy C:\tmp\OpenCL.dll C:\Windows\System32\ && reg add "HKLM\SOFTWARE\Khronos\OpenCL\Vendors" /v "C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409\ocl\bin\x64\intelocl64.dll" /t REG_DWORD /d 0

   For example, run the `Hello Classification Python` sample with the following command:

   .. code-block:: sh

      omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp GPU


@endsphinxdirective

### Build a Docker image manually

**Step 1**: Preparing a Dockerfile

Use the [available Dockerfiles on GitHub](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your settings via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) which can generate a Dockerfile, build, test and deploy an image with the Intel® Distribution of OpenVINO™ toolkit.

**Step 2**: Configuring the Docker Image for Different Devices

@sphinxdirective

.. tab:: Installing Additional Dependencies for CPU

   1. Installing CMake

   To add CMake to the image, add the following commands to the Dockerfile:

   .. code-block:: sh

      RUN powershell.exe -Command `
      Invoke-WebRequest -URI https://cmake.org/files/v3.14/cmake-3.14.7-win64-x64.msi -OutFile %TMP%\\cmake-3.14.7-win64-x64.msi ; `
      Start-Process %TMP%\\cmake-3.14.7-win64-x64.msi -ArgumentList '/quiet /norestart' -Wait ; `
      Remove-Item %TMP%\\cmake-3.14.7-win64-x64.msi -Force

      RUN SETX /M PATH "C:\Program Files\CMake\Bin;%PATH%"
   
   In case of proxy issues, add the `ARG HTTPS_PROXY` and `-Proxy %%HTTPS_PROXY%` settings to the `powershell.exe` command to the Dockerfile. Then build a Docker image:

   .. code-block:: sh
      docker build . -t <image_name> `
      --build-arg HTTPS_PROXY=<https://your_proxy_server:port>   

   2. Installing Microsoft Visual Studio Build Tools

   You may add Microsoft Visual Studio Build Tools to a Windows OS Docker image using the [offline](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019) or [online](https://docs.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2019) installers for Build Tools.

   Microsoft Visual Studio Build Tools are licensed as a supplement your existing Microsoft Visual Studio license. Any images built with these tools should be for your personal use or for use in your organization in accordance with your existing Visual Studio and Windows licenses.

   To add MSBuild 2019 to the image, add the following commands to the Dockerfile:

   .. code-block:: sh
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

   In case of proxy issues, use the [offline installer for Build Tools](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2019).

.. tab:: Configuring the Image for GPU

   .. note::
      Since GPU is not supported in prebuilt images or [default Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles), you must make sure the Additional Requirements for GPU in <a href="#system-requirements">System Requirements</a> are met, and do the following steps to build the image manually.

   1. Reuse one of [available Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles). You may also use your own Dockerfile.

   2. Check your [Windows host and container isolation process compatibility](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility).

   3. Find the appropriate Windows container base image on [DockerHub](https://hub.docker.com/_/microsoft-windows) and set up your host/container version in the `FROM` Dockerfile instruction.
      
   For example, in the `openvino_c_dev_<version>.dockerfile`, change:

   .. code-block:: sh 
   
      FROM mcr.microsoft.com/windows/servercore:ltsc2019 AS ov_base
   
   to:

   .. code-block:: sh
   
      FROM mcr.microsoft.com/windows:20H2
   
   4. Build the Docker image by running the following command:

   .. code-block:: sh

      docker build --build-arg package_url=<OpenVINO pkg> -f <Dockerfile> -t <image_name> .
   
   5. Copy `OpenCL.dll` from your `C:\Windows\System32` host folder to any `temp` directory:

   .. code-block:: sh

      mkdir C:\tmp
      copy C:\Windows\System32\OpenCL.dll C:\tmp


@endsphinxdirective

**Step 3**: Running the Docker Image on Different Devices

@sphinxdirective

.. tab:: Running the Image on CPU

   To start the interactive session, run the following command:
   
   .. code-block:: sh

      docker run -it --rm <image_name>
   

   If you want to try some samples, run the image with the following command:

   .. code-block:: sh

      docker run -it --rm <image_name> 
      cmd /S /C "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp CPU"
   

.. tab:: Running the Image on GPU

   .. note::
      Since GPU is not supported in prebuilt images or [default Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles), you must make sure the Additional Requirements for GPU in <a href="#system-requirements">System Requirements</a> are met, and configure and build the image manually before you may run inferences on a GPU.


   1. To try inference on a GPU, run the image with the following command:

   .. code-block:: sh
   
      docker run -it --rm -u ContainerAdministrator --isolation process --device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599 -v C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409:C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409 -v C:\tmp:C:\tmp <image_name>

   where:

   - `--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599` is a reserved interface class GUID for a GPU device.
   - `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409` is the path to OpenCL driver home directory. To find it on your PC, run the `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*` regular expression.
   - `C:\tmp` is the folder with the copy of `OpenCL.dll` from your `C:\Windows\System32` host folder.

   2. Copy `OpenCL.dll` to the `C:\Windows\System32` folder inside the container and set appropriate registry entry. Now, you may run inference on a GPU device:

   .. code-block:: sh
   
      copy C:\tmp\OpenCL.dll C:\Windows\System32\ && reg add "HKLM\SOFTWARE\Khronos\OpenCL\Vendors" /v "C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_518f2921ba495409\ocl\bin\x64\intelocl64.dll" /t REG_DWORD /d 0

   For example, run the `Hello Classification Python` sample with the following command:

   .. code-block:: sh

      omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -kO https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python samples\python\hello_classification\hello_classification.py public\googlenet-v1\FP16\googlenet-v1.xml car_1.bmp GPU


@endsphinxdirective

## Additional Resources

   - [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You may reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.
   - Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)