# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

This guide provides steps on creating a Docker image with Intel® Distribution of OpenVINO™ toolkit for Linux* and using the image on different devices. 

There are two options to install OpenVINO with Docker:

* You can <a href="#using-prebuilt-image">get a prebilt image and run the image directly</a>;
* Or <a href="#build-image-manually">build a docker image manually and run the image on different devices</a>.

## System requirements

@sphinxdirective
.. tab:: Target Operating Systems

  * Ubuntu\* 18.04 long-term support (LTS), 64-bit
  * Ubuntu\* 20.04 long-term support (LTS), 64-bit
  * Red Hat\* Enterprise Linux 8 (64 bit)

.. tab:: Host Operating Systems

  * Linux
  * Windows Subsystem for Linux 2 on CPU or GPU
  * macOS on CPU only

@endsphinxdirective


## <a name="using-prebuilt-image"></a>Installing OpenVINO with a prebuilt image

### Step 1: Get a prebuilt image from provided sources

You can find prebuilt images on:

- [Docker Hub](https://hub.docker.com/u/openvino)
- [Red Hat* Quay.io](https://quay.io/organization/openvino)
- [Red Hat* Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3)
- [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino)

### Step 2: Run the image

Run the Docker image using the following command:
```
docker run -it --rm <image_name>
```

## <a name="build-image-manually"></a>Installing OpenVINO by building a Docker image manually

### Step 1: Create a dockerfile

You can use the [available Dockerfiles on GitHub](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your setting via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) which can generate a Dockerfile, build, test, and deploy an image with the the Intel® Distribution of OpenVINO™ toolkit.
You can also try our [Tutorials](https://github.com/openvinotoolkit/docker_ci/tree/master/docs/tutorials) which demonstrate the usage of Docker containers with OpenVINO. 

### Step 2: Configure and run the image for different devices

#### Using Docker image on CPU

Note the following things:

- Kernel reports the same information for all containers as for native application, for example, CPU, memory information.
- All instructions that are available to host process available for process in container, including, for example, AVX2, AVX512. No restrictions.
- Docker\* does not use virtualization or emulation. The process in Docker* is just a regular Linux process, but it is isolated from external world on kernel level. Performance loss is minor.

To use the OpenVINO Docker image on CPU, you don’t need extra configurations. Run the Docker image with the following command:
```
docker run -it --rm <image_name>
```

#### Using Docker image on GPU

> **NOTE**: Only Intel® integrated graphics are supported.

**Prerequisites**

- GPU is not available in container by default, you must attach it to the container.
- Kernel driver must be installed on the host.
- Intel® OpenCL™ runtime package must be included into the container.
- In the container, non-root user must be in the `video` and `render` groups. To add a user to the render group, follow the [Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu* 20.04](https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md).

To launch a Linux image on Windows Subsystem for Linux 2, note the following things:

- Only Windows 10 with 21H2 update or above installed and Windows 11 are supported.
- Intel GPU driver on Windows host with version 30.0.100.9684 or above need be installed. Please see [this article](https://www.intel.com/content/www/us/en/artificial-intelligence/harness-the-power-of-intel-igpu-on-your-machine.html#articleparagraph_983312434) for more details.
- The Docker images for 2022.1 release contain preinstalled recommended version of OpenCL Runtime with WSL2 support.

##### Step 1: Configure the image for GPU

If you have installed your custom version of GPU driver and want to build an image, see the following examples for your Dockerfile:

**Ubuntu 18.04/20.04**:

```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino

RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb" --output "intel-gmmlib_19.3.2_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb" --output "intel-igc-core_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb" --output "intel-igc-opencl_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb" --output "intel-opencl_19.41.14441_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.41.14441_amd64.deb" --output "intel-ocloc_19.04.12237_amd64.deb" && \
    dpkg -i /tmp/opencl/*.deb && \
    ldconfig && \
    rm /tmp/opencl
```

or you can use the installation script `install_NEO_OCL_driver.sh` if you previously installed OpenVINO in the Dockerfile. The installation script will install the recommended version of GPU driver for the operating system in the image.

```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino

WORKDIR ${INTEL_OPENVINO_DIR}/install_dependencies
RUN ./install_NEO_OCL_driver.sh --no_numa -y && \
    rm -rf /var/lib/apt/lists/*
```

**RHEL 8**:

```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino
RUN groupmod -g 44 video

RUN yum update -y && yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    yum update -y && yum install -y ocl-icd ocl-icd-devel && \
    yum clean all && rm -rf /var/cache/yum && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-19.3.2-1.el7.x86_64.rpm/download -o intel-gmmlib-19.3.2-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm/download -o intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-core-1.0.2597-1.el7.x86_64.rpm/download -o intel-igc-core-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm/download -o intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm/download -o  intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-opencl-19.41.14441-1.el7.x86_64.rpm/download -o intel-opencl-19.41.14441-1.el7.x86_64.rpm \
    rpm -ivh ${TEMP_DIR}/*.rpm && \
    ldconfig && \
    rm -rf ${TEMP_DIR} && \
    yum remove -y epel-release
```

or you can use the installation script `install_NEO_OCL_driver.sh` if you previously installed OpenVINO in the Dockerfile. The installation script will install the recommended version of GPU driver for the operating system in the image.

```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino
RUN groupmod -g 44 video

WORKDIR ${INTEL_OPENVINO_DIR}/install_dependencies
RUN ./install_NEO_OCL_driver.sh --no_numa -y  && \
    yum clean all && rm -rf /var/cache/yum && \
    yum remove -y epel-release
```

##### Step 2: Run the image on GPU

To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:

* Ubuntu 18 or RHEL8:
    ```sh
    docker run -it --rm --device /dev/dri <image_name>
    ```

* WSL2:
    ```sh
    docker run -it --rm --device /dev/dxg --volume /usr/lib/wsl:/usr/lib/wsl <image_name>
    ```

> **NOTE**: If your host system is Ubuntu 20, follow the [Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu* 20.04](https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md). 

#### Using Docker image on Intel® Neural Compute Stick 2

**Known limitations:**

- Intel® Neural Compute Stick 2 device changes its VendorID and DeviceID during execution and each time looks for a host system as a brand new device. It means it cannot be mounted as usual.
- UDEV events are not forwarded to the container by default it does not know about device reconnection.
- Only one NCS2 device connected to the host can be used when running inference in a container.

Use the following steps for Intel® Neural Compute Stick 2:

1. Get rid of UDEV by rebuilding `libusb` without UDEV support in the Docker* image by adding the following commands to a `Dockerfile`:
   - **Ubuntu 18.04/20.04**:
        ```sh
        ARG BUILD_DEPENDENCIES="autoconf \
                        automake \
                        build-essential \
                        libtool \
                        unzip \
                        udev"
        RUN apt-get update && \
            apt-get install -y --no-install-recommends ${BUILD_DEPENDENCIES} && \
            rm -rf /var/lib/apt/lists/*

        WORKDIR /opt
        RUN curl -L https://github.com/libusb/libusb/archive/v1.0.22.zip --output v1.0.22.zip && \
            unzip v1.0.22.zip && rm -rf v1.0.22.zip

        WORKDIR /opt/libusb-1.0.22
        RUN ./bootstrap.sh && \
            ./configure --disable-udev --enable-shared && \
            make -j4

        WORKDIR /opt/libusb-1.0.22/libusb
        RUN /bin/mkdir -p '/usr/local/lib' && \
            /bin/bash ../libtool --mode=install /usr/bin/install -c   libusb-1.0.la '/usr/local/lib' && \
            /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
            /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
            /bin/mkdir -p '/usr/local/lib/pkgconfig'

        WORKDIR /opt/libusb-1.0.22/
        RUN /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
            cp /opt/intel/openvino/runtime/3rdparty/97-myriad-usbboot.rules /etc/udev/rules.d/ && \
            ldconfig
        ```

2. Run the Docker image:
    ```sh
    docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
    ```

##### An alternative option to run the image

While the steps above is not working, you can also run container in the privileged mode, enable the Docker network configuration as host, and mount all devices to the container. Run the following command:
```sh
docker run -it --rm --privileged -v /dev:/dev --network=host <image_name>
```

> **NOTE**: This option is not recommended, as conflicts with Kubernetes* and other tools that use orchestration and private networks may occur. Please use it with caution and only for troubleshooting purposes.


#### Using Docker image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

##### Step 1: Configure the image

> **NOTE**: When building the Docker image, create a user in the Dockerfile that has the same UID (User Identifier) and GID (Group Identifier) as the user which that runs hddldaemon on the host, and then run the application in the Docker image with this user. This step is necessary to run the container as a non-root user.


To use the Docker container for inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, do the following steps:

1. Set up the environment on the host machine to be used for running Docker*. It is required to execute hddldaemon, which is responsible for communication between the HDDL plugin and the board. To learn how to set up the environment (the OpenVINO package or HDDL package must be pre-installed), see [Configuration guide for HDDL device](https://github.com/openvinotoolkit/docker_ci/blob/master/install_guide_vpu_hddl.md) or [Configurations for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux](configurations-for-vpu-linux.md).
2. Prepare the Docker image by adding the following commands to a Dockerfile:
   - **Ubuntu 18.04**:
        ```sh
        WORKDIR /tmp
        RUN apt-get update && \
            apt-get install -y --no-install-recommends \
                libboost-filesystem1.65-dev \
                libboost-thread1.65-dev \
                libjson-c3 libxxf86vm-dev && \
        rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
        ```   
   - **Ubuntu 20.04**:
        ```sh
        WORKDIR /tmp
        RUN apt-get update && \
            apt-get install -y --no-install-recommends \
                libboost-filesystem-dev \
                libboost-thread-dev \
                libjson-c4 \
                libxxf86vm-dev && \
        rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
        ```   
3. Run `hddldaemon` on the host in a separate terminal session using the following command:
    ```sh
    $HDDL_INSTALL_DIR/hddldaemon
    ```

##### Step 2: Run the image

To run the built Docker* image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, use the following command:

```sh
docker run -it --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp <image_name>
```

> **NOTE**: The device `/dev/ion` needs to be shared to be able to use ion buffers among the plugin, `hddldaemon` and the kernel.


> **NOTE**: Since separate inference tasks share the same HDDL service communication interface (the service creates mutexes and a socket file in `/var/tmp`), `/var/tmp` needs to be mounted and shared among them.



**If the ion driver is not enabled**

In some cases, the ion driver is not enabled (for example, due to a newer kernel version or iommu (Input-Output Memory Management Unit) incompatibility). `lsmod | grep myd_ion` returns empty output. To resolve this issue, use the following command:
```sh
docker run -it --rm --net=host -v /var/tmp:/var/tmp –-ipc=host <image_name>
```
If that still does not solve the issue, try starting `hddldaemon` with the root user on host. However, this approach is not recommended. Please use with caution.

## Running samples in OpenVINO Docker image

To run the `Hello Classification Sample` on a specific inference device, run the following commands:

**CPU**:

```sh
docker run -it --rm <image_name>
/bin/bash -c "cd ~ && omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 /opt/intel/openvino/samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp CPU"
```

**GPU**:

```sh
docker run -itu root:root  --rm --device /dev/dri:/dev/dri <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp GPU"
```

**MYRIAD**:

```sh
docker run -itu root:root --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp MYRIAD"
```

**HDDL**:

```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp HDDL"
```

## Additional resources

- [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.
- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- Intel® Neural Compute Stick 2 Get Started: [https://software.intel.com/en-us/neural-compute-stick/get-started](https://software.intel.com/en-us/neural-compute-stick/get-started)
